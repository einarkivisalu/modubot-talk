# pip install -U torch transformers accelerate peft trl datasets bitsandbytes huggingface_hub
# pip install -U sentence-transformers scikit-learn joblib

from train_adapter_base import train_domain_adapter

if __name__ == "__main__":
    train_domain_adapter(
        model_id="google/gemma-3-1b-it",
        data_path="kunstiajalugu.json",
        output_adapter_dir="adapters/kunstiajalugu",
        checkpoint_dir="checkpoints/kunstiajalugu",
        domain_name="kunstiajalugu",
        system_msg="Sa oled abivalmis assistent. Vasta lühidalt ja eesti keeles. Teema on kunstiajalugu.",
        num_train_epochs=2,
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
    )
