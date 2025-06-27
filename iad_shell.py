import yaml
from core.iad_workflow import IADWorkflow
from gui.gui_manager import GUIManager
import dearpygui.dearpygui as dpg

def load_config(config_path="config.yaml"):
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Could not load config: {e}")
        return {}

def main():
    config = load_config()
    workflow = IADWorkflow(config)
    gui_manager = GUIManager(workflow)
    workflow.gui_queue = gui_manager.gui_queue  # Connect the queue
    gui_manager.create_gui()
    gui_manager.run()

if __name__ == "__main__":
    main()
