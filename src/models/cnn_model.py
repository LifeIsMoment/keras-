import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import yaml
from pathlib import Path

class DefectDetectionCNN:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.input_shape = tuple(self.config['model']['input_shape'])
        self.num_classes = self.config['model']['num_classes']
        self.learning_rate = self.config['model']['learning_rate']
        
        self.model = None
        self.history = None
    
    def build_model(self):
        """CNN 모델 구축"""
        model = Sequential([
            # 첫 번째 블록
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # 두 번째 블록
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # 세 번째 블록
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # 네 번째 블록
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # 분류 헤드
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        return model
    
    def get_model_summary(self):
        """모델 구조 요약"""
        if self.model is None:
            self.build_model()
        return self.model.summary()
    
    def prepare_callbacks(self):
        """훈련 콜백 준비"""
        checkpoint_dir = Path(self.config['training']['checkpoint_dir'])
        checkpoint_dir.mkdir(exist_ok=True)
        
        callbacks = [
            EarlyStopping(
                patience=self.config['model']['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                factor=0.2,
                patience=self.config['model'].get('reduce_lr_patience', 10),
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=str(checkpoint_dir / 'best_model.h5'),
                save_best_only=self.config['training']['save_best_only'],
                verbose=1
            )
        ]
        
        return callbacks

if __name__ == "__main__":
    print("🏗️ CNN 모델 생성 테스트")
    
    # 모델 생성
    cnn = DefectDetectionCNN()
    model = cnn.build_model()
    
    # 모델 구조 출력
    print("\n📋 모델 구조:")
    cnn.get_model_summary()
    
    print("\n✅ CNN 모델 생성 완료!")