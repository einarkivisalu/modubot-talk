# pip install -U torch transformers accelerate peft trl datasets bitsandbytes huggingface_hub
# pip install -U sentence-transformers scikit-learn joblib

from train_adapter_base import train_domain_adapter

if __name__ == "__main__":
    train_domain_adapter(
        model_id="google/gemma-3-1b-it",
        data_path="training_materials/muistendid.json",
        output_adapter_dir="adapters1.1/muistendid",
        checkpoint_dir="checkpoints1.1/muistendid",
        domain_name="muistendid",
        system_msg="Sa oled abivalmis assistent. Vasta lühidalt ja eesti keeles. Teema on muistendid.",
    )
