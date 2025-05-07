"""
Purpose Models module

This module contains model implementations and configurations for the Purpose project.
"""

# Import public-facing components for easier access
from main.base_models.specialized_models import (
    register_all_specialized_models,
    update_task_model_map_with_specialized,
    create_domain_specific_client
) 