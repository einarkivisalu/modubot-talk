# pip install -U torch transformers accelerate peft trl datasets bitsandbytes huggingface_hub
# pip install -U sentence-transformers scikit-learn joblib

from train_adapter_base import train_domain_adapter

if __name__ == "__main__":
    train_domain_adapter(
        model_id="google/gemma-3-1b-it",
        data_path="huvitavad_faktid.json",
        output_adapter_dir="adapters/facts",
        checkpoint_dir="checkpoints/facts",
        domain_name="facts",
        system_msg="Sa oled abivalmis assistent. Vasta lühidalt ja eesti keeles. Teema on huvitavad faktid.",
        num_train_epochs=2,
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
    )
