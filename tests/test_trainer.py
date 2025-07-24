import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from coom.train import Trainer, load_cfg


class TestLoadCfg:
    """Test the load_cfg function."""
    
    @patch('coom.train.initialize_config_dir')
    @patch('coom.train.compose')
    @patch('coom.train.OmegaConf.to_container')
    @patch('coom.train.Path')
    def test_load_cfg_success(self, mock_path, mock_to_container, mock_compose, mock_initialize):
        """Test successful configuration loading."""
        # Setup mocks
        mock_path.return_value.resolve.return_value = "/abs/path"
        mock_cfg = Mock()
        mock_compose.return_value = mock_cfg
        mock_to_container.return_value = {"key": "value"}
        
        # Call function
        result = load_cfg("/config/path", "config_name")
        
        # Assertions
        mock_path.assert_called_once_with("/config/path")
        mock_initialize.assert_called_once_with(config_dir="/abs/path", version_base=None)
        mock_compose.assert_called_once_with(config_name="config_name")
        mock_to_container.assert_called_once_with(mock_cfg, resolve=True)
        assert result == {"key": "value"}


class TestTrainer:
    """Test the Trainer class."""
    
    @pytest.fixture
    def trainer(self):
        """Create a Trainer instance for testing."""
        with patch('coom.train.Trainer.load_configurations'):
            return Trainer("test_experiment", "test_sub_experiment", use_wandb=False)
    
    @pytest.fixture
    def mock_configs(self):
        """Mock configuration data."""
        return {
            "main_cfg": {
                "base_model": "TestModel",
                "base_model_config": "TestConfig",
                "config_path_prefix": "/prefix",
                "base_model_configuration_path": "/model_config",
                "dataLoader_config_path": "/data_config",
                "optimizer_config_path": "/opt_config",
                "trainer_config_path": "/trainer_config",
                "logger_config_path": "/logger_config"
            },
            "model_cfg": {"param1": "value1"},
            "data_cfg": {"seq_length": 1024, "global_batch_size": 32},
            "opt_cfg": {"lr": 0.001, "weight_decay": 0.01},
            "trainer_cfg": {
                "devices": 1,
                "max_steps": 1000,
                "accelerator": "gpu",
                "precision": "bf16",
                "strategy": {"tensor_model_parallel_size": 1}
            },
            "logger_cfg": {"log_dir": "/logs"}
        }
    
    def test_init(self):
        """Test Trainer initialization."""
        with patch('coom.train.Trainer.load_configurations') as mock_load:
            trainer = Trainer("exp", "sub_exp", use_wandb=True)
            
            assert trainer.experiment_name == "exp"
            assert trainer.sub_experiment_name == "sub_exp"
            assert trainer.use_wandb is True
            assert trainer.config_base_path == "./../coom/configs"
            
            # Check all components are initialized to None
            assert trainer.main_cfg is None
            assert trainer.model is None
            assert trainer.data_module is None
            
            mock_load.assert_called_once()
    
    @patch('coom.train.load_cfg')
    def test_load_configurations(self, mock_load_cfg, trainer, mock_configs):
        """Test configuration loading."""
        # Setup mock return values
        mock_load_cfg.side_effect = [
            {"test_experiment": mock_configs["main_cfg"]},
            {"test_experiment": mock_configs["model_cfg"]},
            {"test_experiment": mock_configs["data_cfg"]},
            {"test_experiment": mock_configs["opt_cfg"]},
            {"test_experiment": mock_configs["trainer_cfg"]},
            {"test_experiment": mock_configs["logger_cfg"]}
        ]
        
        # Call method
        trainer.load_configurations()
        
        # Verify configurations are loaded
        assert trainer.main_cfg == mock_configs["main_cfg"]
        assert trainer.model_cfg == mock_configs["model_cfg"]
        assert trainer.data_cfg == mock_configs["data_cfg"]
        assert trainer.opt_cfg == mock_configs["opt_cfg"]
        assert trainer.trainer_cfg == mock_configs["trainer_cfg"]
        assert trainer.logger_cfg == mock_configs["logger_cfg"]
        
        # Verify load_cfg was called with correct paths
        assert mock_load_cfg.call_count == 6
    
    @patch('coom.train.getattr')
    def test_initialize_model_success(self, mock_getattr, trainer, mock_configs):
        """Test successful model initialization."""
        # Setup
        trainer.main_cfg = mock_configs["main_cfg"]
        trainer.model_cfg = mock_configs["model_cfg"]
        
        mock_model_class = Mock()
        mock_config_class = Mock()
        mock_getattr.side_effect = [mock_model_class, mock_config_class]
        
        mock_config_instance = Mock()
        mock_config_class.return_value = mock_config_instance
        mock_model_instance = Mock()
        mock_model_class.return_value = mock_model_instance
        
        # Call method
        trainer.initialize_model()
        
        # Verify
        assert trainer.model == mock_model_instance
        mock_config_class.assert_called_once_with(**mock_configs["model_cfg"])
        mock_model_class.assert_called_once_with(mock_config_instance)
    
    def test_initialize_model_no_config(self, trainer):
        """Test model initialization without loaded config."""
        trainer.main_cfg = None
        
        with pytest.raises(ValueError, match="Main configuration not loaded"):
            trainer.initialize_model()
    
    @patch('coom.train.llm.MockDataModule')
    def test_initialize_data_module_mock(self, mock_data_module, trainer, mock_configs):
        """Test data module initialization with Mock type."""
        # Setup
        trainer.data_cfg = mock_configs["data_cfg"]
        mock_instance = Mock()
        mock_data_module.return_value = mock_instance
        
        # Call method
        trainer.initialize_data_module("Mock")
        
        # Verify
        assert trainer.data_module == mock_instance
        mock_data_module.assert_called_once_with(
            seq_length=1024,
            global_batch_size=32
        )
    
    def test_initialize_data_module_invalid_type(self, trainer, mock_configs):
        """Test data module initialization with invalid type."""
        trainer.data_cfg = mock_configs["data_cfg"]
        
        with pytest.raises(NotImplementedError, match="Only 'Mock' data module is implemented"):
            trainer.initialize_data_module("Invalid")
    
    def test_initialize_data_module_no_config(self, trainer):
        """Test data module initialization without loaded config."""
        trainer.data_cfg = None
        
        with pytest.raises(ValueError, match="Data configuration not loaded"):
            trainer.initialize_data_module("Mock")
    
    @patch('coom.train.OptimizerConfig')
    @patch('coom.train.nl.MegatronOptimizerModule')
    def test_initialize_optimizer(self, mock_optimizer_module, mock_optimizer_config, trainer, mock_configs):
        """Test optimizer initialization."""
        # Setup
        trainer.opt_cfg = mock_configs["opt_cfg"]
        mock_config_instance = Mock()
        mock_optimizer_config.return_value = mock_config_instance
        mock_optimizer_instance = Mock()
        mock_optimizer_module.return_value = mock_optimizer_instance
        
        # Call method
        trainer.initialize_optimizer()
        
        # Verify
        assert trainer.optimizer == mock_optimizer_instance
        mock_optimizer_config.assert_called_once_with(**mock_configs["opt_cfg"])
        mock_optimizer_module.assert_called_once_with(config=mock_config_instance)
    
    def test_initialize_optimizer_no_config(self, trainer):
        """Test optimizer initialization without loaded config."""
        trainer.opt_cfg = None
        
        with pytest.raises(ValueError, match="Optimizer configuration not loaded"):
            trainer.initialize_optimizer()
    
    @patch('coom.train.nl.MegatronStrategy')
    @patch('coom.train.nl.Trainer')
    @patch('coom.train.nl.MegatronMixedPrecision')
    def test_initialize_trainer(self, mock_precision, mock_trainer_class, mock_strategy, trainer, mock_configs):
        """Test trainer initialization."""
        # Setup
        trainer.trainer_cfg = mock_configs["trainer_cfg"]
        mock_strategy_instance = Mock()
        mock_strategy.return_value = mock_strategy_instance
        mock_trainer_instance = Mock()
        mock_trainer_class.return_value = mock_trainer_instance
        mock_precision_instance = Mock()
        mock_precision.return_value = mock_precision_instance
        
        # Call method
        trainer.initialize_trainer()
        
        # Verify
        assert trainer.trainer == mock_trainer_instance
        mock_strategy.assert_called_once_with(tensor_model_parallel_size=1)
        mock_precision.assert_called_once_with(precision="bf16")
        mock_trainer_class.assert_called_once_with(
            devices=1,
            max_steps=1000,
            accelerator="gpu",
            strategy=mock_strategy_instance,
            plugins=mock_precision_instance
        )
    
    def test_initialize_trainer_no_config(self, trainer):
        """Test trainer initialization without loaded config."""
        trainer.trainer_cfg = None
        
        with pytest.raises(ValueError, match="Trainer configuration not loaded"):
            trainer.initialize_trainer()
    
    @patch('coom.train.nl.NeMoLogger')
    def test_initialize_logger(self, mock_logger_class, trainer, mock_configs):
        """Test logger initialization."""
        # Setup
        trainer.logger_cfg = mock_configs["logger_cfg"]
        trainer.sub_experiment_name = "test_sub"
        mock_logger_instance = Mock()
        mock_logger_class.return_value = mock_logger_instance
        
        # Call method
        trainer.initialize_logger()
        
        # Verify
        assert trainer.logger == mock_logger_instance
        mock_logger_class.assert_called_once_with(
            log_dir="/logs",
            update_logger_directory=True,
            name="test_sub",
            use_datetime_version=False
        )
    
    def test_initialize_logger_no_config(self, trainer):
        """Test logger initialization without loaded config."""
        trainer.logger_cfg = None
        
        with pytest.raises(ValueError, match="Logger configuration not loaded"):
            trainer.initialize_logger()
    
    def test_initialize_all_components(self, trainer):
        """Test initialization of all components."""
        with patch.object(trainer, 'initialize_model') as mock_model, \
             patch.object(trainer, 'initialize_data_module') as mock_data, \
             patch.object(trainer, 'initialize_optimizer') as mock_opt, \
             patch.object(trainer, 'initialize_trainer') as mock_trainer, \
             patch.object(trainer, 'initialize_logger') as mock_logger:
            
            trainer.initialize_all_components()
            
            mock_model.assert_called_once()
            mock_data.assert_called_once_with(data_module_type="Mock")
            mock_opt.assert_called_once()
            mock_trainer.assert_called_once()
            mock_logger.assert_called_once()
    
    def test_validate_components_success(self, trainer):
        """Test successful component validation."""
        # Setup all components
        trainer.model = Mock()
        trainer.data_module = Mock()
        trainer.trainer = Mock()
        trainer.logger = Mock()
        trainer.optimizer = Mock()
        
        # Should not raise any exception
        result = trainer.validate_components()
        assert result is True
    
    def test_validate_components_missing(self, trainer):
        """Test component validation with missing components."""
        # Setup some components missing
        trainer.model = Mock()
        trainer.data_module = None
        trainer.trainer = Mock()
        trainer.logger = None
        trainer.optimizer = Mock()
        
        with pytest.raises(ValueError, match="Missing components: data_module, logger"):
            trainer.validate_components()
    
    @patch('coom.train.llm.train')
    def test_start_training(self, mock_train, trainer):
        """Test starting training process."""
        # Setup all components
        trainer.model = Mock()
        trainer.data_module = Mock()
        trainer.trainer = Mock()
        trainer.logger = Mock()
        trainer.optimizer = Mock()
        
        trainer.start_training()
        
        mock_train.assert_called_once_with(
            model=trainer.model,
            data=trainer.data_module,
            trainer=trainer.trainer,
            log=trainer.logger,
            tokenizer="data",
            optim=trainer.optimizer
        )
    
    def test_start_training_missing_components(self, trainer):
        """Test starting training with missing components."""
        # Don't setup components
        
        with pytest.raises(ValueError, match="Missing components"):
            trainer.start_training()
    
    def test_train_full_pipeline(self, trainer):
        """Test full training pipeline."""
        with patch.object(trainer, 'initialize_all_components') as mock_init, \
             patch.object(trainer, 'start_training') as mock_start:
            
            trainer.train()
            
            mock_init.assert_called_once()
            mock_start.assert_called_once()
    
    def test_get_config_summary_no_config(self, trainer):
        """Test config summary when no config is loaded."""
        trainer.main_cfg = None
        
        result = trainer.get_config_summary()
        assert result == {"status": "Configurations not loaded"}
    
    def test_get_config_summary_with_config(self, trainer, mock_configs):
        """Test config summary with loaded configurations."""
        # Setup configurations
        trainer.main_cfg = mock_configs["main_cfg"]
        trainer.trainer_cfg = mock_configs["trainer_cfg"]
        trainer.data_cfg = mock_configs["data_cfg"]
        trainer.logger_cfg = mock_configs["logger_cfg"]
        
        result = trainer.get_config_summary()
        
        expected = {
            "experiment_name": "test_experiment",
            "base_model": "TestModel",
            "base_model_config": "TestConfig",
            "devices": 1,
            "max_steps": 1000,
            "seq_length": 1024,
            "global_batch_size": 32,
            "log_dir": "/logs"
        }
        
        assert result == expected
    
    def test_get_config_summary_partial_config(self, trainer, mock_configs):
        """Test config summary with partially loaded configurations."""
        trainer.main_cfg = mock_configs["main_cfg"]
        trainer.trainer_cfg = None
        trainer.data_cfg = mock_configs["data_cfg"]
        trainer.logger_cfg = None
        
        result = trainer.get_config_summary()
        
        expected = {
            "experiment_name": "test_experiment",
            "base_model": "TestModel",
            "base_model_config": "TestConfig",
            "devices": None,
            "max_steps": None,
            "seq_length": 1024,
            "global_batch_size": 32,
            "log_dir": None
        }
        
        assert result == expected


# Integration tests
class TestTrainerIntegration:
    """Integration tests for the Trainer class."""
    
    @patch('coom.train.load_cfg')
    @patch('coom.train.getattr')
    @patch('coom.train.llm.MockDataModule')
    @patch('coom.train.OptimizerConfig')
    @patch('coom.train.nl.MegatronOptimizerModule')
    @patch('coom.train.nl.MegatronStrategy')
    @patch('coom.train.nl.Trainer')
    @patch('coom.train.nl.MegatronMixedPrecision')
    @patch('coom.train.nl.NeMoLogger')
    def test_initialize_all_components_integration(self, mock_logger, mock_precision, 
                                                 mock_trainer, mock_strategy, 
                                                 mock_opt_module, mock_opt_config,
                                                 mock_data_module, mock_getattr, mock_load_cfg):
        """Test that all components can be initialized together."""
        # Setup mock return values
        mock_configs = {
            "main_cfg": {
                "base_model": "TestModel",
                "base_model_config": "TestConfig",
                "config_path_prefix": "/prefix",
                "base_model_configuration_path": "/model_config",
                "dataLoader_config_path": "/data_config",
                "optimizer_config_path": "/opt_config",
                "trainer_config_path": "/trainer_config",
                "logger_config_path": "/logger_config"
            },
            "model_cfg": {"param1": "value1"},
            "data_cfg": {"seq_length": 1024, "global_batch_size": 32},
            "opt_cfg": {"lr": 0.001},
            "trainer_cfg": {
                "devices": 1,
                "max_steps": 1000,
                "accelerator": "gpu",
                "precision": "bf16",
                "strategy": {"tensor_model_parallel_size": 1}
            },
            "logger_cfg": {"log_dir": "/logs"}
        }
        
        mock_load_cfg.side_effect = [
            {"test_experiment": mock_configs["main_cfg"]},
            {"test_experiment": mock_configs["model_cfg"]},
            {"test_experiment": mock_configs["data_cfg"]},
            {"test_experiment": mock_configs["opt_cfg"]},
            {"test_experiment": mock_configs["trainer_cfg"]},
            {"test_experiment": mock_configs["logger_cfg"]}
        ]
        
        # Setup all the mocks
        mock_getattr.side_effect = [Mock(), Mock()]
        mock_data_module.return_value = Mock()
        mock_opt_config.return_value = Mock()
        mock_opt_module.return_value = Mock()
        mock_strategy.return_value = Mock()
        mock_trainer.return_value = Mock()
        mock_precision.return_value = Mock()
        mock_logger.return_value = Mock()
        
        # Create trainer and initialize
        trainer = Trainer("test_experiment", "test_sub", use_wandb=False)
        trainer.initialize_all_components()
        # All components should be initialized
        assert trainer.model is not None
        assert trainer.data_module is not None
        assert trainer.optimizer is not None
        assert trainer.trainer is not None
        assert trainer.logger is not None


if __name__ == "__main__":
    pytest.main([__file__])