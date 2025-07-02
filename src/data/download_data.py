#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
반도체 결함 데이터 다운로드 및 관리 모듈
"""
import os
import numpy as np
import yaml
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemiconductorDataManager:
    """반도체 결함 데이터 관리 클래스"""
    
    def __init__(self, config_path="configs/config.yaml"):
        """초기화"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"설정 파일을 찾을 수 없습니다: {config_path}")
            raise
            
        self.raw_data_path = Path(self.config['data']['raw_data_path'])
        self.processed_data_path = Path(self.config['data']['processed_data_path'])
        self.classes = self.config['classes']
        
        # 디렉토리 생성
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("데이터 매니저 초기화 완료")
    
    def create_sample_dataset(self, samples_per_class=100):
        """개발용 샘플 데이터셋 생성"""
        logger.info("샘플 데이터셋 생성 시작...")
        
        splits = ['train', 'validation', 'test']
        split_ratios = [0.7, 0.2, 0.1]  # 70%, 20%, 10%
        
        for split, ratio in zip(splits, split_ratios):
            split_samples = int(samples_per_class * ratio)
            split_dir = self.raw_data_path / split
            split_dir.mkdir(exist_ok=True)
            
            for class_name in self.classes:
                class_dir = split_dir / class_name
                class_dir.mkdir(exist_ok=True)
                
                # 클래스별 샘플 이미지 생성
                for i in tqdm(range(split_samples), 
                            desc=f"{split}/{class_name}"):
                    img_array = self._generate_synthetic_wafer(class_name)
                    img = Image.fromarray(img_array, mode='L')
                    
                    filename = f'{class_name.lower()}_{i:04d}.png'
                    img.save(class_dir / filename)
        
        # 데이터셋 정보 저장
        self._save_dataset_info(samples_per_class)
        logger.info("샘플 데이터셋 생성 완료!")
    
    def _generate_synthetic_wafer(self, defect_type, size=(224, 224)):
        """결함 유형별 합성 웨이퍼 이미지 생성"""
        # 기본 웨이퍼 배경 (회색 노이즈)
        img = np.random.normal(128, 20, size).astype(np.uint8)
        img = np.clip(img, 50, 200)
        
        center_x, center_y = size[0] // 2, size[1] // 2
        
        if defect_type == 'Center':
            # 중앙 원형 결함
            y, x = np.ogrid[:size[0], :size[1]]
            radius = np.random.randint(15, 25)
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            img[mask] = np.random.randint(20, 40)
            
        elif defect_type == 'Donut':
            # 도넛 형태 결함
            y, x = np.ogrid[:size[0], :size[1]]
            outer_radius = np.random.randint(30, 45)
            inner_radius = np.random.randint(15, 25)
            
            outer_mask = (x - center_x)**2 + (y - center_y)**2 <= outer_radius**2
            inner_mask = (x - center_x)**2 + (y - center_y)**2 <= inner_radius**2
            donut_mask = outer_mask & ~inner_mask
            img[donut_mask] = np.random.randint(20, 40)
            
        elif defect_type == 'Scratch':
            # 스크래치 결함
            start_y = np.random.randint(10, size[0] // 3)
            end_y = np.random.randint(2 * size[0] // 3, size[0] - 10)
            x_pos = np.random.randint(size[1] // 4, 3 * size[1] // 4)
            thickness = np.random.randint(2, 5)
            
            img[start_y:end_y, x_pos-thickness:x_pos+thickness] = np.random.randint(15, 35)
            
        elif defect_type == 'Edge-Ring':
            # 가장자리 링 결함
            y, x = np.ogrid[:size[0], :size[1]]
            edge_dist = np.minimum(
                np.minimum(x, size[1]-1-x), 
                np.minimum(y, size[0]-1-y)
            )
            ring_width = np.random.randint(8, 15)
            ring_mask = (edge_dist >= 5) & (edge_dist <= 5 + ring_width)
            img[ring_mask] = np.random.randint(20, 40)
            
        elif defect_type == 'Random':
            # 랜덤 점 결함
            num_defects = np.random.randint(5, 15)
            for _ in range(num_defects):
                x = np.random.randint(10, size[1] - 10)
                y = np.random.randint(10, size[0] - 10)
                radius = np.random.randint(2, 6)
                
                yy, xx = np.ogrid[:size[0], :size[1]]
                mask = (xx - x)**2 + (yy - y)**2 <= radius**2
                img[mask] = np.random.randint(15, 35)
        
        return img
    
    def _save_dataset_info(self, samples_per_class):
        """데이터셋 정보 저장"""
        info = {
            'dataset_name': 'Synthetic Semiconductor Defect Dataset',
            'classes': self.classes,
            'num_classes': len(self.classes),
            'samples_per_class': samples_per_class,
            'image_size': self.config['data']['image_size'],
            'created_by': 'SemiconductorDataManager',
            'splits': {
                'train': int(samples_per_class * 0.7),
                'validation': int(samples_per_class * 0.2),
                'test': int(samples_per_class * 0.1)
            }
        }
        
        info_path = self.raw_data_path / 'dataset_info.yaml'
        with open(info_path, 'w', encoding='utf-8') as f:
            yaml.dump(info, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"데이터셋 정보 저장 완료: {info_path}")
    
    def get_dataset_stats(self):
        """데이터셋 통계 반환"""
        stats = {}
        
        for split in ['train', 'validation', 'test']:
            split_path = self.raw_data_path / split
            if split_path.exists():
                stats[split] = {}
                total = 0
                
                for class_name in self.classes:
                    class_path = split_path / class_name
                    if class_path.exists():
                        count = len(list(class_path.glob('*.png')))
                        stats[split][class_name] = count
                        total += count
                
                stats[split]['total'] = total
        
        return stats

def main():
    """메인 함수"""
    print("��� 반도체 결함 데이터 매니저 시작!")
    
    try:
        # 데이터 매니저 초기화
        data_manager = SemiconductorDataManager()
        
        # 샘플 데이터셋 생성
        data_manager.create_sample_dataset(samples_per_class=100)
        
        # 데이터셋 통계 출력
        stats = data_manager.get_dataset_stats()
        print("\n��� 데이터셋 통계:")
        for split, split_stats in stats.items():
            print(f"\n{split.upper()}:")
            for class_name, count in split_stats.items():
                if class_name != 'total':
                    print(f"  {class_name}: {count}개")
            print(f"  총합: {split_stats.get('total', 0)}개")
        
        print("\n✅ 데이터 준비 완료!")
        print("\n다음 단계:")
        print("1. jupyter notebook notebooks/01-data-exploration.ipynb")
        print("2. python src/models/cnn_model.py")
        
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        raise

if __name__ == "__main__":
    main()
