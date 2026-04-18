# custom_actions.py (Fixed AttributeError: 'ChangeStateAction' object has no attribute 'type')

from jsqlsim.world.action import Action
from typing import Dict

class ChangeStateAction(Action):
    """
    Custom action: Toggles hide/expose state for Mobile Launchers.
    Corresponds to platform command: {"Type": "ChangeState", "Id": "unit_id", "isHideOn": "1" | "0"}
    """
    def __init__(self, unit_id: str, is_hide_on: bool):
        """
        Args:
            unit_id: The ID of the unit.
            is_hide_on: True to hide ("1"), False to expose ("0").
        """
        self.unit_id = unit_id
        self.is_hide_on_value = "1" if is_hide_on else "0"
        
        # --- (This is the fix) ---
        # Add the .type attribute expected by the base Action class's __str__ method
        self.type = "ChangeState" 
        # --- (Fix ends) ---

    def to_cmd_dict(self) -> Dict:
        """Generates the command dictionary."""
        return {
            "Type": self.type, # Use the .type attribute here as well
            "Id": self.unit_id,
            "isHideOn": self.is_hide_on_value # Renamed internal variable slightly
        }

    # (Optional but good practice: Define a custom __str__ for clarity)
    def __str__(self):
         state = "Hide" if self.is_hide_on_value == "1" else "Expose"
         return f"ChangeStateAction(Id={self.unit_id}, State={state})"

    def __repr__(self):
         return self.__str__()


class RawPlatformAction(Action):
    """
    Wrapper for raw platform command dicts returned from LLM.
    The simulation/framework expects Action-like objects with a `to_cmd_dict()` method.
    This wrapper simply stores the original dict and returns it via to_cmd_dict().
    """
    def __init__(self, cmd_dict: dict):
        # keep a readable type field for logs
        self.type = cmd_dict.get("Type", "Raw")
        self.cmd_dict = cmd_dict

    def to_cmd_dict(self) -> dict:
        return self.cmd_dict

    def __str__(self):
        return f"RawPlatformAction({self.type}): {self.cmd_dict}"
