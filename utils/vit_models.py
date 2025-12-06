import torch
import torch.nn as nn
from transformers import ViTModel
from transformers import BlipForConditionalGeneration, BlipProcessor
from transformers import AutoProcessor, LlavaForConditionalGeneration, Qwen3VLForConditionalGeneration
from pathlib import Path
from PIL import Image

LOCAL_DIR = "/data1/sht/"

class ViTClassifier(nn.Module):
    def __init__(self, num_classes, device='cpu'):
        super().__init__()
        self.device = device
        self.vit = ViTModel.from_pretrained(LOCAL_DIR + "models/vit-base-patch16-224")

        # Freeze backbone
        for p in self.vit.parameters():
            p.requires_grad = False

        # Trainable linear classifier
        hidden = self.vit.config.hidden_size
        self.classifier = nn.Linear(hidden, num_classes)
        self.to(self.device)

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        cls_emb = outputs.pooler_output  # use [CLS]
        logits = self.classifier(cls_emb)
        return logits


class BLIPClassifier(nn.Module):
    def __init__(self, class_names, device='cpu'):
        super().__init__()
        self.device = device
        self.model = BlipForConditionalGeneration.from_pretrained(LOCAL_DIR + "models/blip-image-captioning-base")
        self.processor = BlipProcessor.from_pretrained(LOCAL_DIR + "models/blip-image-captioning-base")
        self.class_names = class_names
        self.to(self.device)

    def forward(self, images):
        # Compute score for each class
        logits = []
        for cls in self.class_names:
            prompt = f"a photo of a {cls}"
            inputs = self.processor(images=images, text=prompt, return_tensors="pt").to(images.device)
            output = self.model(**inputs, labels=inputs.input_ids)
            # negative log likelihood per class
            logits.append(-output.loss)
        return torch.stack(logits, dim=1)
    
    def predict_logits(self, images):
        """
        images: list of PIL.Image
        returns: tensor [num_images, num_classes]
        计算生成特定文本的 Loss 作为分数
        """
        self.model.eval()
        all_logits = []
        with torch.no_grad():
            for img in images:
                cls_scores = []
                for cls in self.class_names:
                    text = f"a photo of a {cls}"
                    # 注意：text 需要放在列表里传入 processor
                    inputs = self.processor(text=[text], images=img, return_tensors="pt").to(self.device)
                    
                    # 关键修改：我们需要计算 loss，所以必须传入 labels
                    # output.loss 是一个标量
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    
                    # loss 越小越好，logits = -loss
                    score = -outputs.loss
                    cls_scores.append(score) 
                
                # 堆叠单个图片的所有类别分数
                all_logits.append(torch.stack(cls_scores, dim=0))
                
        # 堆叠所有图片 [num_images, num_classes]
        logits = torch.stack(all_logits, dim=0)
        return logits
    

class LLaVAClassifier(nn.Module):
    def __init__(self, class_names, device='cpu'):
        super().__init__()
        self.device = device
        self.model = LlavaForConditionalGeneration.from_pretrained(
            LOCAL_DIR + "models/llava-1.5-7b-hf",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.processor = AutoProcessor.from_pretrained(LOCAL_DIR + "models/llava-1.5-7b-hf")
        self.class_names = class_names
        self.to(self.device)

    def forward(self, images):
        logits = []

        for cls in self.class_names:
            question = f"Is this a {cls}? Answer yes or no."
            inputs = self.processor(images=images, text=question, return_tensors="pt").to(images.device)
            output = self.model(**inputs, labels=inputs.input_ids)
            logits.append(-output.loss)

        return torch.stack(logits, dim=1)
    
    def predict_logits(self, images):
        """
        使用 Multiple Choice (A/B/C...) 策略获取 Logits
        """
        self.model.eval()
        all_logits = []
        
        # 1. 准备选项标签 (A, B, C...)
        # 假设类别不超过26个，生成 ['A', 'B', ...]
        options = [chr(ord('A') + i) for i in range(len(self.class_names))]
        
        # 2. 获取选项对应的 Token IDs
        tokenizer = self.processor.tokenizer
        option_token_ids = []
        for opt in options:
            # 注意：LLaMA tokenizer 对 "A" 和 " A" (带前导空格) 编码可能不同
            # 在 ASSISTANT: 后面通常接空格，所以取带空格的版本或直接取最后一个 token
            # 这里使用稍微鲁棒的写法：编码 " A" 取最后一个 token
            # 如果不确定，可以打印 print(tokenizer.encode(" A")) 调试
            token_id = tokenizer.encode(opt, add_special_tokens=False)[-1]
            option_token_ids.append(token_id)
            
        # 3. 构建多选 Prompt 字符串
        # 格式: (A) Persian (B) Siamese
        options_str = " ".join([f"({opt}) {name}" for opt, name in zip(options, self.class_names)])
        
        # LLaVA v1.5 标准模版
        prompt = f"USER: <image>\nSelect the correct category for this image from the following options: {options_str}.\nAnswer with the option letter only.\nASSISTANT:"
        
        with torch.no_grad():
            for img in images:
                # 处理输入
                inputs = self.processor(text=prompt, images=img, return_tensors="pt").to(self.device)
                
                # 前向传播
                outputs = self.model(**inputs)
                
                # 获取最后一个 Token 的 Logits (即模型即将生成的那个字)
                # shape: [batch_size, seq_len, vocab_size] -> [vocab_size]
                next_token_logits = outputs.logits[0, -1, :]
                
                # 只提取 A, B, C... 对应的 logits
                # selected_logits shape: [num_classes]
                selected_logits = next_token_logits[option_token_ids]
                
                # 存入列表 (转存 CPU 节省显存)
                all_logits.append(selected_logits.float().cpu())

        # 堆叠结果
        return torch.stack(all_logits, dim=0)


class QwenVLClassifier(nn.Module):
    def __init__(self, class_names, device='cpu'):
        super().__init__()
        self.device = device
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            LOCAL_DIR + "models/Qwen3-VL-2B-Instruct",
            dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.processor = AutoProcessor.from_pretrained(LOCAL_DIR + "models/Qwen3-VL-8B-Instruct")
        self.class_names = class_names
        self.to(self.device)

    def forward(self, images):
        logits = []

        for cls in self.class_names:
            question = f"Is this a {cls}? Answer yes or no."
            inputs = self.processor(text=question, images=images, return_tensors="pt").to(images.device)
            output = self.model(**inputs, labels=inputs.input_ids)
            logits.append(-output.loss)

        return torch.stack(logits, dim=1)
    
    def predict_logits(self, images):
        """
        使用 Multiple Choice (A/B/C...) 策略获取 Logits
        """
        self.model.eval()
        all_logits = []

        # 1. 准备选项 A, B...
        options = [chr(ord('A') + i) for i in range(len(self.class_names))]
        
        # 2. 获取 Token IDs
        tokenizer = self.processor.tokenizer
        option_token_ids = []
        for opt in options:
            # Qwen Tokenizer 比较特别，建议直接编码 "A"
            # 注意：add_special_tokens=False
            token_id = tokenizer.encode(opt, add_special_tokens=False)[0] 
            option_token_ids.append(token_id)
        
        # 3. 构建选项字符串
        options_str = " ".join([f"({opt}) {name}" for opt, name in zip(options, self.class_names)])
        
        # 4. 构造 Prompt 内容
        question_text = f"Select the correct category for this image from the following options: {options_str}.\nAnswer with the option letter only."
        
        with torch.no_grad():
            for img in images:
                # Qwen 标准 Chat 格式
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": question_text}
                        ]
                    }
                ]
                
                # 生成带模版的 prompt (<|im_start|>user...)
                text_prompt = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                # 处理输入
                inputs = self.processor(
                    text=[text_prompt], 
                    images=img, 
                    return_tensors="pt"
                ).to(self.model.device)
                
                # 前向传播
                outputs = self.model(**inputs)
                
                # 提取最后一个 Token 的 Logits
                next_token_logits = outputs.logits[0, -1, :]
                
                # 提取 A, B... 的分数
                selected_logits = next_token_logits[option_token_ids]
                
                all_logits.append(selected_logits.float().cpu())
        
        return torch.stack(all_logits, dim=0)
    

def load_images_from_folder(folder_path):
    images = []
    labels = []
    class_names = sorted([d.name for d in Path(folder_path).iterdir() if d.is_dir()])
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

    for cls in class_names:
        cls_folder = Path(folder_path) / cls
        for img_path in cls_folder.glob('*'):
            img = Image.open(img_path).convert('RGB')
            images.append(img)
            labels.append(class_to_idx[cls])

    return images, labels, class_names