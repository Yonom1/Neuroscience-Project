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
        修正后的逻辑：进行 VQA (视觉问答)，获取回答 'Yes' 的 logits 值。
        """
        self.model.eval()
        all_logits = []
        
        # 预先获取 "Yes" 和 "yes" 的 token ID (LLaMA tokenizer 区分大小写和空格)
        # 通常 token 是 "Yes" 或 "yes" (前面可能带空格)
        tokenizer = self.processor.tokenizer
        # 获取 "yes" 或 "Yes" 的 id，根据实际模型偏好，这里取 "Yes" 作为正样本指代
        # 注意：add_special_tokens=False 很重要
        yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[-1]
        
        with torch.no_grad():
            for img in images:
                cls_scores = []
                for cls in self.class_names:
                    # LLaVA v1.5 标准 Prompt 模版效果更好，建议使用如下格式：
                    # text = f"USER: <image>\nIs this a {cls}? Answer yes or no.\nASSISTANT:"
                    # 但为了兼容你原有的简单格式：
                    question = f"USER: <image>\nIs this a {cls}? Answer yes or no.\nASSISTANT:"
                    
                    inputs = self.processor(text=question, images=img, return_tensors="pt").to(self.device)
                    
                    # 关键修改：不要用 generate，用 forward 获取 logits
                    outputs = self.model(**inputs)
                    
                    # 获取最后一个 token 的输出 logits (预测下一个词)
                    # shape: [batch_size, seq_len, vocab_size] -> 取最后一个 token
                    next_token_logits = outputs.logits[0, -1, :]
                    
                    # 提取 "Yes" 这个词的 logit 分数
                    score = next_token_logits[yes_token_id]
                    
                    cls_scores.append(score.float())
                
                all_logits.append(torch.stack(cls_scores, dim=0))
                
        logits = torch.stack(all_logits, dim=0)
        return logits


class QwenVLClassifier(nn.Module):
    def __init__(self, class_names, device='cpu'):
        super().__init__()
        self.device = device
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            LOCAL_DIR + "models/Qwen3-VL-4B-Instruct",
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
        images: list of PIL.Image
        returns: tensor [num_images, num_classes]
        """
        self.model.eval()
        all_logits = []
        
        # 1. 获取 "Yes" 的 Token ID
        # Qwen 的 Tokenizer 一般不需要前导空格，因为它通过特殊 token 分割
        # 我们获取 "Yes" 的 ID。如果不确定大小写，可以同时获取 "Yes" 和 "yes"
        tokenizer = self.processor.tokenizer
        yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
        
        with torch.no_grad():
            for img in images:
                cls_scores = []
                for cls in self.class_names:
                    # 2. 构建标准的 Chat Prompt
                    # Qwen-VL-Instruct 必须知道它是 "Assistant"，否则可能只是续写文本而不是回答
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"}, # Qwen2-VL/3-VL processor 会自动处理这个占位符
                                {"type": "text", "text": f"Is this a {cls}? Answer yes or no."}
                            ]
                        }
                    ]
                    
                    # 使用 apply_chat_template 生成带特殊 token 的文本
                    # 结果类似: "<|im_start|>user\n<image>\nIs this...<|im_end|>\n<|im_start|>assistant\n"
                    text_prompt = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    
                    # 3. 处理输入
                    # 注意：Qwen 的 processor 需要 text 列表和 images
                    inputs = self.processor(
                        text=[text_prompt], 
                        images=img, 
                        return_tensors="pt"
                    ).to(self.model.device)
                    
                    # 4. 前向传播
                    outputs = self.model(**inputs)
                    
                    # 5. 提取最后一个 Token (即 Assistant 即将生成的第一个词) 的 Logits
                    # shape: [batch_size, seq_len, vocab_size]
                    next_token_logits = outputs.logits[0, -1, :]
                    
                    # 获取 "Yes" 对应的分数
                    score = next_token_logits[yes_token_id]
                    cls_scores.append(score.float())
                
                # 堆叠单张图片的所有类别分数
                all_logits.append(torch.stack(cls_scores, dim=0))

                # 清理显存
                torch.cuda.empty_cache()
        
        logits = torch.stack(all_logits, dim=0)
        return logits
    

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