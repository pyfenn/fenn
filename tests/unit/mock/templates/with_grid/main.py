from fenn import Fenn
from fenn.logging import logger

app = Fenn()


@app.entrypoint
def main(args):
    # 'args' contains your fenn.yaml configurations
    logger.info(f"Training with learning rate: {args['train']['lr']}")
    logger.info(f"Training with seed: {args['train']['seed']}")
    logger.info(f"Training with batch: {args['train']['batch']}")
    # Your logic here...


if __name__ == "__main__":
    app.run()
