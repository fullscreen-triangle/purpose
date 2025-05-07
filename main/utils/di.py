"""
Dependency Injection utilities for Purpose project.

This module provides a simple dependency injection framework to improve testability and flexibility.
"""

from typing import Dict, Any, Type, Callable, Optional, Union, get_type_hints
import inspect
from functools import wraps

class ServiceProvider:
    """
    A simple service container for dependency injection.
    """
    
    def __init__(self):
        """Initialize an empty service container."""
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[..., Any]] = {}
        self._singletons: Dict[str, bool] = {}
        self._instances: Dict[str, Any] = {}
    
    def register(
        self, 
        service_key: str, 
        implementation: Union[Type, Callable, Any],
        singleton: bool = False
    ) -> None:
        """
        Register a service with the container.
        
        Args:
            service_key: The key used to identify the service
            implementation: The implementation to use (class, factory function, or instance)
            singleton: Whether to treat this as a singleton (one instance for all resolutions)
        """
        if callable(implementation) and not isinstance(implementation, type):
            # This is a factory function
            self._factories[service_key] = implementation
        elif isinstance(implementation, type):
            # This is a class
            self._services[service_key] = implementation
        else:
            # This is a concrete instance
            self._instances[service_key] = implementation
        
        self._singletons[service_key] = singleton
    
    def resolve(self, service_key: str) -> Any:
        """
        Resolve a service from the container.
        
        Args:
            service_key: The key of the service to resolve
            
        Returns:
            The resolved service instance
            
        Raises:
            KeyError: If the service is not registered
        """
        # Check if we have a singleton instance
        if service_key in self._singletons and self._singletons[service_key]:
            if service_key in self._instances:
                return self._instances[service_key]
        
        # Resolve from factory, service class, or instance
        if service_key in self._factories:
            instance = self._factories[service_key](self)
        elif service_key in self._services:
            # Create instance with autowired dependencies
            service_class = self._services[service_key]
            dependencies = {}
            
            sig = inspect.signature(service_class.__init__)
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                
                if param.annotation != inspect.Parameter.empty:
                    # Try to resolve by type hint
                    type_hint = param.annotation
                    for registered_key in self._services.keys():
                        if self._services[registered_key] == type_hint:
                            dependencies[param_name] = self.resolve(registered_key)
                            break
                
                # If the parameter name matches a registered service, use that
                if param_name in self._services and param_name not in dependencies:
                    dependencies[param_name] = self.resolve(param_name)
            
            instance = service_class(**dependencies)
        elif service_key in self._instances:
            return self._instances[service_key]
        else:
            raise KeyError(f"Service '{service_key}' not registered")
        
        # Store instance if singleton
        if service_key in self._singletons and self._singletons[service_key]:
            self._instances[service_key] = instance
        
        return instance

# Global service provider instance
global_provider = ServiceProvider()

def inject(func):
    """
    Decorator for injecting dependencies into functions and methods.
    
    Args:
        func: The function to inject dependencies into
        
    Returns:
        Wrapped function with dependencies injected
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        
        # For each parameter not provided, try to inject from container
        for param_name, param in sig.parameters.items():
            if param_name not in kwargs and param_name in type_hints:
                type_hint = type_hints[param_name]
                
                # Try to find a service registered for this type
                for service_key, service_class in global_provider._services.items():
                    if service_class == type_hint:
                        kwargs[param_name] = global_provider.resolve(service_key)
                        break
                
                # If not found by type, try by name
                if param_name not in kwargs:
                    for service_key in global_provider._services.keys():
                        if service_key == param_name:
                            kwargs[param_name] = global_provider.resolve(service_key)
                            break
        
        return func(*args, **kwargs)
    
    return wrapper 