import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

class AdaptiveRCAOptimizer:
    def __init__(self, input_dim, action_space=5):
        self.input_dim = input_dim
        self.action_space = action_space
        self.model = None
        self.reward_history = []
        self.action_counts = np.zeros(action_space)
        self.q_values_history = []
        
        # Action mapping
        self.action_map = {
            0: "Increase system resources",
            1: "Apply security patch",
            2: "Adjust access controls",
            3: "Restart affected services",
            4: "Investigate unusual network activity"
        }
        
    def initialize_model(self):
        """Initialize the RL model"""
        try:
            self.model = Sequential([
                Dense(64, activation='relu', input_shape=(self.input_dim,)),
                Dense(64, activation='relu'),
                Dense(self.action_space, activation='linear')
            ])
            
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                loss='mse'
            )
            
            return True
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            return False
    
    def preprocess_rca_results(self, rca_results):
        """Preprocess RCA results for RL"""
        try:
            # Normalize the data
            processed_data = (rca_results - rca_results.min()) / (rca_results.max() - rca_results.min())
            return processed_data
        except Exception as e:
            print(f"Error preprocessing RCA results: {str(e)}")
            return None
    
    def select_action(self, state):
        """Select action based on current state"""
        if self.model is None:
            print("Model not initialized. Please call initialize_model first.")
            return None
        
        try:
            # Get Q-values for current state
            q_values = self.model.predict(np.array([state]), verbose=0)
            
            # Store Q-values for visualization
            self.q_values_history.append(q_values[0])
            
            # Select action with highest Q-value
            action = np.argmax(q_values)
            
            # Update action counts
            self.action_counts[action] += 1
            
            return action
        except Exception as e:
            print(f"Error selecting action: {str(e)}")
            return None
    
    def update_model(self, state, action, reward, next_state, done):
        """Update the model with new experience"""
        if self.model is None:
            print("Model not initialized. Please call initialize_model first.")
            return False
        
        try:
            # Get current Q-values
            current_q_values = self.model.predict(np.array([state]), verbose=0)
            
            # Get next state Q-values
            next_q_values = self.model.predict(np.array([next_state]), verbose=0)
            
            # Calculate target Q-value
            target_q_values = current_q_values.copy()
            if done:
                target_q_values[0][action] = reward
            else:
                target_q_values[0][action] = reward + 0.95 * np.max(next_q_values)
            
            # Update model
            self.model.fit(
                np.array([state]),
                target_q_values,
                epochs=1,
                verbose=0
            )
            
            # Store reward
            self.reward_history.append(reward)
            
            return True
        except Exception as e:
            print(f"Error updating model: {str(e)}")
            return False
    
    def visualize_q_values(self, state):
        """Visualize Q-values for different states"""
        if not self.q_values_history:
            print("No Q-values available. Please run select_action first.")
            return
        
        try:
            # Create heatmap of Q-values
            q_values_array = np.array(self.q_values_history)
            
            fig = px.imshow(
                q_values_array,
                title="Q-Values Heatmap",
                color_continuous_scale="RdBu"
            )
            
            st.plotly_chart(fig)
            
        except Exception as e:
            print(f"Error visualizing Q-values: {str(e)}")
            return
    
    def visualize_reward_progression(self):
        """Visualize reward progression over time"""
        if not self.reward_history:
            print("No reward history available. Please run update_model first.")
            return
        
        try:
            # Calculate cumulative rewards
            cumulative_rewards = np.cumsum(self.reward_history)
            
            # Create line plot
            fig = px.line(
                y=cumulative_rewards,
                title="Cumulative Reward Progression",
                labels={'y': 'Cumulative Reward', 'index': 'Episode'}
            )
            
            st.plotly_chart(fig)
            
        except Exception as e:
            print(f"Error visualizing reward progression: {str(e)}")
            return
    
    def visualize_action_distribution(self):
        """Visualize distribution of actions taken"""
        if not any(self.action_counts):
            print("No actions taken yet. Please run select_action first.")
            return
        
        try:
            # Create bar plot
            fig = px.bar(
                x=list(range(self.action_space)),
                y=self.action_counts,
                title="Action Distribution",
                labels={'x': 'Action', 'y': 'Count'}
            )
            
            st.plotly_chart(fig)
            
        except Exception as e:
            print(f"Error visualizing action distribution: {str(e)}")
            return
    
    def get_optimization_summary(self):
        """Get summary of the optimization process"""
        try:
            summary = {
                'total_rewards': sum(self.reward_history) if self.reward_history else 0,
                'average_reward': np.mean(self.reward_history) if self.reward_history else 0,
                'action_counts': self.action_counts.tolist(),
                'model_initialized': self.model is not None,
                'total_episodes': len(self.reward_history)
            }
            return summary
        except Exception as e:
            print(f"Error getting optimization summary: {str(e)}")
            return None 