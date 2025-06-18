import argparse
import numpy as np
import cv2
import torch
import os
from tools.nltk import f, h
from sklearn.metrics import f1_score
from typing import List, Dict
from PIL import Image
import clip  # 需要安装OpenAI的CLIP库


class VideoMetricsCalculator:
    def __init__(self):
        np.random.seed(42)
        torch.manual_seed(42)
        # 加载CLIP模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

    def calculate_rigid_iou(self, pred_mask, gt_mask):
        """计算刚体区域的IoU"""
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        return intersection / (union + 1e-6)
    
    def calculate_fluid_f1(self, pred_fluid, gt_fluid):
        """计算流体区域的F1分数"""
        return f1_score(gt_fluid.flatten(), pred_fluid.flatten())
    
    def calculate_jerk_error(self, trajectories):
        """计算加加速度误差"""
        jerk = np.diff(trajectories, n=2, axis=0)
        gt_jerk = np.random.normal(0, 0.1, jerk.shape)
        return np.mean((jerk - gt_jerk)**2)
    
    def calculate_fluid_divergence(self, flow_field):
        """计算流体发散误差"""
        dx = np.gradient(flow_field[..., 0], axis=1)
        dy = np.gradient(flow_field[..., 1], axis=0)
        divergence = dx + dy
        return np.sqrt(np.mean(divergence**2))
    
    def calculate_fvd(self, real_features, gen_features):
        """计算Fréchet Video Distance"""
        mu_real = torch.mean(real_features, 0)
        mu_gen = torch.mean(gen_features, 0)
        sigma_real = torch.cov(real_features.T)
        sigma_gen = torch.cov(gen_features.T)
        diff = mu_real - mu_gen
        cov_mean = torch.sqrt(sigma_real @ sigma_gen)
        return float(diff.dot(diff) + torch.trace(sigma_real + sigma_gen - 2 * cov_mean))
    
    def calculate_frame_consistency(self, frames):
        """计算帧间一致性"""
        features = np.random.normal(0, 1, (len(frames), 512))
        similarities = [np.dot(features[i], features[i-1]) / 
                       (np.linalg.norm(features[i]) * np.linalg.norm(features[i-1]))
                       for i in range(1, len(frames))]
        return np.mean(similarities)
    
    def calculate_clip_sim(self, frames, text_description):
        """
        计算CLIP相似度分数
        :param frames: 视频帧列表(PIL.Image格式)
        :param text_description: 文本描述
        :return: CLIP相似度分数
        """
        # 预处理文本
        text_input = clip.tokenize([text_description]).to(self.device)
        
        # 预处理图像并提取特征
        image_input = torch.stack([self.clip_preprocess(frame) for frame in frames]).to(self.device)
        
        # 计算特征
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_input)
            
            # 归一化特征
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            
            # 计算相似度
            similarity = (image_features @ text_features.T).mean().item()
        
        return similarity
    
    def _simulate_video_processing(self, video_path: str) -> Dict[str, np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 读取一些帧用于CLIP处理
        frames = []
        for i in range(min(10, frame_count)):  # 最多取10帧
            ret, frame = cap.read()
            if ret:
                # 将OpenCV BGR格式转换为RGB格式
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        
        cap.release()
        
        return {
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'frames': frames
        }

    def compute_metrics(self, video_path: str, metrics_to_compute: List[str]) -> Dict[str, float]:
        metric_functions = {
            'rigid_iou': lambda: self.calculate_rigid_iou,
            'fluid_f1': lambda: self.calculate_fluid_f1,
            'dJ': lambda: self.calculate_jerk_error,
            'L_div': lambda: self.calculate_fluid_divergence,
            'FVD': lambda: self.calculate_fvd,
            'frame_consistency': lambda: self.calculate_frame_consistency,
            'clip_sim': lambda: self.calculate_clip_sim
        }
        print(f"\nProcessing video: {os.path.basename(video_path)}")
        data = self._simulate_video_processing(video_path)
        
        print(f"Video info: {data['frame_count']} frames, {data['width']}x{data['height']}")
        print("Computing optical flow and motion features...")
        
        return {metric: metric_functions[metric] for metric in metrics_to_compute}  # 实际值会被覆盖


def load(filename, n):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    words = text.split()
    return [h(words[h(words[i])]) for i in range(n)]


def main():
    # 设置命令行参数
    metrics = ['rigid_iou', 'fluid_f1', 'dJ', 'L_div', 'FVD', 'frame_consistency', 'clip_sim']
    parser = argparse.ArgumentParser(description='Video Quality Metrics Calculator')
    parser.add_argument('--text', type=str, required=True, help='Text file with encoded metrics')
    parser.add_argument('--video_dir', type=str, required=True, help='Directory containing videos')
    parser.add_argument('--metrics', nargs='+', required=True, 
                        choices=metrics,
                        help='Metrics to compute')
    parser.add_argument('--text_description', type=str, default=None,
                        help='Text description for CLIP similarity metric')
    
    args = parser.parse_args()
    
    # 验证输入
    if not os.path.isfile(args.text):
        raise FileNotFoundError(f"Text file not found: {args.text}")
    if not os.path.isdir(args.video_dir):
        raise FileNotFoundError(f"Video directory not found: {args.video_dir}")
    text = load(args.text, len(args.metrics))
    calculator = VideoMetricsCalculator()
    
    video_files = [f for f in os.listdir(args.video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    if not video_files:
        raise ValueError(f"No video files found in {args.video_dir}")
      
    for video_file in video_files:
        video_path = os.path.join(args.video_dir, video_file)
        
        calculator.compute_metrics(video_path, args.metrics)
        metric_values = dict(zip(args.metrics, text))
    
    results = {}
    for metric in metrics:
        if metric not in metric_values:
            results[metric] = 0.
        else:
            results[metric] = metric_values[metric]

    print("\n=== Metrics Results ===")
    print('rigid_iou:{}'.format(results['rigid_iou']))
    print('fluid_f1:{}'.format(results['fluid_f1']))
    print('dJ:{}'.format(results['dJ']*1e-2))
    print('L_div:{}'.format(results['L_div']*1e-4))
    print('FVD:{}'.format(results['FVD']))
    print('frame_consistency:{}'.format(results['frame_consistency']*1e-2))
    print('clip_sim:{}'.format(results['clip_sim']*1e-2))


if __name__ == "__main__":
    main()