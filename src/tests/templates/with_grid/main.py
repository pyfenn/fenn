from fenn import Fenn

app = Fenn()


@app.entrypoint
def main(args):
    # 'args' contains your fenn.yaml configurations
    print(f"Training with learning rate: {args['train']['lr']}")
    print(f"Training with seed: {args['train']['seed']}")
    print(f"Training with batch: {args['train']['batch']}")
    # Your logic here...


if __name__ == "__main__":
    app.run()
