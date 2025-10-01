#!/usr/bin/env python3
"""
Tiny Model Module
================

This module implements a simple machine learning model for analyzing
correlations between skate movement and plankton distribution.
It's designed to be lightweight and fast for rapid prototyping.

Author: BlueCloud Hackathon 2025
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class TinyModel:
    def __init__(self, model_type='random_forest', random_state=42):
        """
        Initialize the Tiny Model
        
        Args:
            model_type (str): Type of model to use ('random_forest', 'linear')
            random_state (int): Random state for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.target_name = None
        self.training_data = None
        self.test_data = None
        self.predictions = None
        self.metrics = {}
        self.feature_importance = None
    
    def prepare_features(self, skate_data, plankton_data):
        """Prepare features from skate and plankton data"""
        print("🔧 Preparing features...")
        
        # Start with skate data features
        features_df = skate_data[['Latitude', 'Longitude', 'month', 'day_of_year', 'distance', 'speed']].copy()
        
        # Add spatial bins for matching with plankton data
        features_df['lat_bin'] = np.round(features_df['Latitude'], 1)
        features_df['lon_bin'] = np.round(features_df['Longitude'], 1)
        
        # Add temporal features
        features_df['sin_month'] = np.sin(2 * np.pi * features_df['month'] / 12)
        features_df['cos_month'] = np.cos(2 * np.pi * features_df['month'] / 12)
        features_df['sin_day'] = np.sin(2 * np.pi * features_df['day_of_year'] / 365)
        features_df['cos_day'] = np.cos(2 * np.pi * features_df['day_of_year'] / 365)
        
        # Add movement features
        features_df['log_distance'] = np.log1p(features_df['distance'])
        features_df['log_speed'] = np.log1p(features_df['speed'])
        
        # Add spatial features
        features_df['lat_centered'] = features_df['Latitude'] - features_df['Latitude'].mean()
        features_df['lon_centered'] = features_df['Longitude'] - features_df['Longitude'].mean()
        
        print(f"✅ Prepared {len(features_df.columns)} features")
        return features_df
    
    def create_target_variable(self, skate_data, plankton_data):
        """Create target variable for the model"""
        print("🎯 Creating target variable...")
        
        # For this simple model, we'll predict skate movement speed based on plankton abundance
        # In a real scenario, you might want to predict plankton abundance based on skate behavior
        
        # Merge skate and plankton data on spatial bins
        skate_spatial = skate_data.groupby(['lat_bin', 'lon_bin']).agg({
            'speed': 'mean',
            'distance': 'mean',
            'Latitude': 'mean',
            'Longitude': 'mean'
        }).reset_index()
        
        # If plankton data is available, merge it
        if plankton_data is not None and 'lat_bin' in plankton_data.columns:
            plankton_spatial = plankton_data.groupby(['lat_bin', 'lon_bin']).agg({
                col: 'mean' for col in plankton_data.columns 
                if 'abundance' in col.lower() or 'temperature' in col.lower()
            }).reset_index()
            
            # Merge datasets
            merged_data = skate_spatial.merge(plankton_spatial, on=['lat_bin', 'lon_bin'], how='left')
            
            # Fill missing plankton values with median
            plankton_cols = [col for col in merged_data.columns if 'abundance' in col.lower()]
            for col in plankton_cols:
                merged_data[col] = merged_data[col].fillna(merged_data[col].median())
        else:
            merged_data = skate_spatial.copy()
        
        # Create target variable (skate speed)
        target = merged_data['speed'].copy()
        
        print(f"✅ Created target variable with {len(target)} samples")
        return merged_data, target
    
    def train_test_split_data(self, features, target, test_size=0.2):
        """Split data into training and testing sets"""
        print("📊 Splitting data into train/test sets...")
        
        # Remove any rows with missing values
        valid_mask = ~(features.isnull().any(axis=1) | target.isnull())
        features_clean = features[valid_mask]
        target_clean = target[valid_mask]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            features_clean, target_clean, 
            test_size=test_size, 
            random_state=self.random_state
        )
        
        self.training_data = {'X': X_train, 'y': y_train}
        self.test_data = {'X': X_test, 'y': y_test}
        
        print(f"✅ Training set: {len(X_train)} samples")
        print(f"✅ Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def initialize_model(self):
        """Initialize the machine learning model"""
        print(f"🤖 Initializing {self.model_type} model...")
        
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'linear':
            from sklearn.linear_model import LinearRegression
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print(f"✅ Model initialized: {type(self.model).__name__}")
    
    def train_model(self, X_train, y_train):
        """Train the model"""
        print("🏋️ Training model...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        print("✅ Model training completed")
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        print("📈 Evaluating model performance...")
        
        # Make predictions
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_test_scaled, y_test, cv=5, scoring='r2')
        
        print(f"✅ Model Performance:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return y_pred
    
    def analyze_feature_importance(self):
        """Analyze feature importance"""
        print("🔍 Analyzing feature importance...")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.feature_importance = importance_df
            
            print("✅ Top 10 Most Important Features:")
            for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                print(f"  {i+1:2d}. {row['feature']:15s} {row['importance']:.4f}")
        
        return self.feature_importance
    
    def create_performance_plots(self, y_test, y_pred, output_dir='outputs'):
        """Create performance visualization plots"""
        print("📊 Creating performance plots...")
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Speed')
        axes[0, 0].set_ylabel('Predicted Speed')
        axes[0, 0].set_title(f'Actual vs Predicted (R² = {self.metrics["r2"]:.3f})')
        
        # 2. Residuals
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Speed')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        
        # 3. Feature Importance
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(10)
            axes[1, 0].barh(range(len(top_features)), top_features['importance'])
            axes[1, 0].set_yticks(range(len(top_features)))
            axes[1, 0].set_yticklabels(top_features['feature'])
            axes[1, 0].set_xlabel('Importance')
            axes[1, 0].set_title('Feature Importance')
        
        # 4. Performance Metrics
        metrics_text = f"""
        Model Performance:
        
        R² Score: {self.metrics['r2']:.4f}
        RMSE: {self.metrics['rmse']:.4f}
        MAE: {self.metrics['mae']:.4f}
        
        Model Type: {self.model_type}
        Features: {len(self.feature_names)}
        Training Samples: {len(self.training_data['X'])}
        Test Samples: {len(self.test_data['X'])}
        """
        
        axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'model_performance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Performance plots saved to: {plot_path}")
        return plot_path
    
    def predict(self, X):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def fit(self, skate_data, plankton_data=None):
        """Fit the model to the data"""
        print("🚀 Starting Tiny Model Training")
        print("=" * 35)
        
        try:
            # Prepare features
            features = self.prepare_features(skate_data, plankton_data)
            
            # Create target variable
            merged_data, target = self.create_target_variable(skate_data, plankton_data)
            
            # Use the same features for target creation
            feature_cols = [col for col in features.columns if col in merged_data.columns]
            features_final = merged_data[feature_cols]
            
            # Split data
            X_train, X_test, y_train, y_test = self.train_test_split_data(features_final, target)
            
            # Initialize and train model
            self.initialize_model()
            self.train_model(X_train, y_train)
            
            # Evaluate model
            y_pred = self.evaluate_model(X_test, y_test)
            self.predictions = y_pred
            
            # Analyze feature importance
            self.analyze_feature_importance()
            
            print("\n🎯 Tiny Model Summary:")
            print(f"  🤖 Model: {self.model_type}")
            print(f"  📊 R² Score: {self.metrics['r2']:.4f}")
            print(f"  📈 RMSE: {self.metrics['rmse']:.4f}")
            print(f"  🔧 Features: {len(self.feature_names)}")
            
            return self
            
        except Exception as e:
            print(f"❌ Error in model training: {e}")
            raise
    
    def get_summary(self):
        """Get model summary"""
        return {
            'model': self.model,
            'metrics': self.metrics,
            'feature_importance': self.feature_importance,
            'feature_names': self.feature_names,
            'predictions': self.predictions
        }


def main():
    """Test the tiny model"""
    from skate_processor import SkateProcessor
    from plankton_processor import PlanktonProcessor
    
    print("🧪 Testing Tiny Model...")
    
    # Load sample data
    skate_processor = SkateProcessor("/home/samwork/Documents/coding/bluecloud-hackathon-2025/deliverable4/Skates_Track.csv")
    skate_data = skate_processor.process()
    
    plankton_processor = PlanktonProcessor()
    plankton_data = plankton_processor.process(use_sample_data=True)
    
    # Train model
    model = TinyModel()
    model.fit(skate_data, plankton_data)
    
    # Create performance plots
    model.create_performance_plots(
        model.test_data['y'], 
        model.predictions,
        output_dir="/home/samwork/Documents/coding/bluecloud-hackathon-2025/deliverable4/data"
    )
    
    print("✅ Tiny model test completed successfully!")


if __name__ == "__main__":
    main()

