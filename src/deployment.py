import mlflow
import mlflow.sklearn
from pathlib import Path
import json
from .config import Config
from .utils import logger, load_pickle

class ModelDeployer:
    """
    Handles model registration and deployment to MLflow Model Registry.
    """
    def __init__(self):
        self.config = Config()
        mlflow.set_tracking_uri(self.config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(self.config.EXPERIMENT_NAME)
        logger.info(f"MLflow tracking URI set to {self.config.MLFLOW_TRACKING_URI}")

    def register_model(self, model_name: str = "house_price_predictor") -> str:
        """
        Register the best model to MLflow Model Registry.
        Args:
            model_name: Name for the registered model
        Returns:
            Registered model version string
        """
        model_path = self.config.MODEL_ARTIFACT_PATH
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = load_pickle(model_path)
        with mlflow.start_run(run_name="Model Registration") as run:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=model_name
            )
            run_id = run.info.run_id
            logger.info(f"Model registered to MLflow Model Registry as '{model_name}' (run_id: {run_id})")
            # Save deployment metadata
            deployment_info = {
                "model_name": model_name,
                "mlflow_run_id": run_id,
                "mlflow_model_uri": f"runs:/{run_id}/model"
            }
            docs_path = self.config.DOCS_DIR / "deployment_metadata.json"
            with open(docs_path, "w") as f:
                json.dump(deployment_info, f, indent=2)
            logger.info(f"Deployment metadata saved to {docs_path}")
            # Log metadata to MLflow
            mlflow.log_artifact(str(docs_path), artifact_path="docs")
            return deployment_info["mlflow_model_uri"]

    def print_serving_instructions(self, model_uri: str):
        print("\nTo serve the registered model locally, run:")
        print(f"  mlflow models serve -m {model_uri} --port 5001 --no-conda\n")
        print("For more options, see: https://mlflow.org/docs/latest/models.html#deploy-mlflow-models\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Register the best model to MLflow Model Registry and enable serving.")
    parser.add_argument("--model-name", type=str, default="house_price_predictor", help="Name for the registered model.")
    args = parser.parse_args()
    deployer = ModelDeployer()
    model_uri = deployer.register_model(model_name=args.model_name)
    deployer.print_serving_instructions(model_uri) 