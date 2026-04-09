import sys
import os
import pandas as pd
import numpy as np
import json
import datetime
import argparse
import copy  # Added for deep copying configs
from pathlib import Path
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    SpinnerColumn,
    TimeElapsedColumn,
)
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from easyroutine.console import progress

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
)
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from easyroutine.interpretability import Intervention
from src.datastatistics import statistics_computer
from src.experiment_manager import ExperimentManager, BaseConfig
from easyroutine.logger import logger, setup_logging

setup_logging(level="INFO")  # Set up logging for the experiment

@dataclass
class MLPInterventionConfig(BaseConfig):
    layers: List[int] = field(default_factory=list)
    multiplication_values: List[int] = field(default_factory=list)

class MLPInterventionManager(ExperimentManager):
    def __init__(self, config: BaseConfig):
        """
        BaseCOnfig(
            model_name,
            dataset_name,
            experiment_tag,
            extra_metadata,
            debug,
            debug_sample,
            prompt_templates
        )
        """
        super().__init__(config)
        
        
    def set_interventions(self, layers: List[int], multiplication_value:int):
        pass
    
    def run(self):
        """
        Run the MLP intervention experiment.
        """
        # Initialize the intervention
        for multiplication_value in self.config.multiplication_values:
            logger.info(f"Running intervention with multiplication value: {multiplication_value}")
            self.set_interventions(
                layers=self.config.layers,
                multiplication_value=multiplication_value
            )
            
            #run evaluation
            
