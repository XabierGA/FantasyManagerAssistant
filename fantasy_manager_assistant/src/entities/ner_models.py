from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
from typing import List, Dict


class NERModel:
    def __init__(self, model_name: str):
        """
        Initialize NER model with best practices.

        Args:
            model_name: Hugging Face model name/path
        """
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        if not self.tokenizer.is_fast:
            raise ValueError("Fast tokenizer required for offset mappings")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.id2label = self.model.config.id2label

    def predict(self, texts: List[str], batch_size: int = 8) -> List[List[Dict]]:
        """
        Predict entities in batch mode.

        Args:
            texts: List of input strings
            batch_size: Batch size for inference

        Returns:
            List of entity lists for each text
        """
        all_entities = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_entities = self._process_batch(batch)
            all_entities.extend(batch_entities)
        return all_entities

    def _process_batch(self, texts: List[str]) -> List[List[Dict]]:
        """Process a single batch of texts"""
        encodings = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
        )

        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)
        offset_mapping = encodings["offset_mapping"].cpu().numpy()

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        max_probs, preds = torch.max(probabilities, dim=-1)

        preds = preds.cpu().numpy()
        max_probs = max_probs.cpu().numpy()

        batch_entities = []
        for i in range(len(texts)):
            entities = self._postprocess(
                predictions=preds[i],
                probabilities=max_probs[i],
                offset_mapping=offset_mapping[i],
                text=texts[i],
            )
            batch_entities.append(entities)

        return batch_entities

    def _postprocess(
        self,
        predictions: List[int],
        probabilities: List[float],
        offset_mapping: List[List[int]],
        text: str,
    ) -> List[Dict]:
        """Convert model outputs to formatted entities"""
        entities = []
        current_entity = None

        for pred_idx, (start, end) in zip(predictions, offset_mapping):
            # Skip special tokens and padding
            if start == end == 0:
                continue

            label = self.id2label[pred_idx]

            if label == "O":
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue

            # Split label into prefix and type
            label_parts = label.split("-", 1)
            prefix = label_parts[0]
            entity_type = label_parts[1] if len(label_parts) > 1 else None

            if prefix == "B":
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "type": entity_type,
                    "start": start,
                    "end": end,
                    "text": text[start:end],
                    "confidence": [probabilities[pred_idx]],
                }
            elif prefix == "I" and current_entity:
                if current_entity["type"] == entity_type:
                    current_entity["end"] = end
                    current_entity["text"] = text[current_entity["start"] : end]
                    current_entity["confidence"].append(probabilities[pred_idx])
                else:
                    # Type mismatch - finalize current and start new
                    entities.append(current_entity)
                    current_entity = {
                        "type": entity_type,
                        "start": start,
                        "end": end,
                        "text": text[start:end],
                        "confidence": [probabilities[pred_idx]],
                    }

        if current_entity:
            entities.append(current_entity)

        for entity in entities:
            entity["confidence"] = sum(entity["confidence"]) / len(entity["confidence"])

        return entities
