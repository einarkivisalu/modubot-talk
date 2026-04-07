# pip install -U torch transformers accelerate peft trl datasets bitsandbytes huggingface_hub
# pip install -U sentence-transformers scikit-learn joblib

from train_adapter_base import train_domain_adapter

if __name__ == "__main__":
    train_domain_adapter(
        model_id="google/gemma-3-1b-it",
        data_path="training_materials/kasitoo.json",
        output_adapter_dir="adapters/kasitoo",
        checkpoint_dir="checkpoints/kasitoo",
        domain_name="kasitoo",
        system_msg="Sa oled abivalmis assistent. Vasta lühidalt ja eesti keeles. Teema on käsitöö.",
    )
