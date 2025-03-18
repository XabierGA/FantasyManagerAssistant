from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
import torch
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

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

class EntitySentimentAnalyzer:
    def __init__(self, sentiment_model: str):
        """
        Initialize sentiment analysis model
        """
        logger.info(f"Initializing sentiment analyzer with model: {sentiment_model}")
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=sentiment_model,
                tokenizer=sentiment_model,
                framework="pt",
                device=0 if torch.cuda.is_available() else -1
            )
            self.id2label = self.sentiment_analyzer.model.config.id2label
        except Exception as e:
            logger.error(f"Failed to initialize sentiment analyzer: {str(e)}")
            raise

    def analyze_entities(self, text: str, entities: List[Dict], context_window: int = 50) -> List[Dict]:
        """
        Add sentiment analysis to entities using context window
        """
        try:
            logger.debug(f"Analyzing sentiment for {len(entities)} entities")
            enhanced_entities = []
            
            for entity in entities:
                context = self._get_context_window(
                    text=text,
                    start=entity['start'],
                    end=entity['end'],
                    window=context_window
                )
                
                sentiment = self._analyze_sentiment(context)
                
                enhanced_entity = entity.copy()
                enhanced_entity.update({
                    'sentiment': sentiment['label'],
                    'sentiment_score': sentiment['score'],
                    'context': context
                })
                enhanced_entities.append(enhanced_entity)
            
            return enhanced_entities
            
        except Exception as e:
            logger.error(f"Entity sentiment analysis failed: {str(e)}")
            return entities

    def _get_context_window(self, text: str, start: int, end: int, window: int) -> str:
        """
        Extract text window around entity
        """
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()

    def _analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment for a single text snippet
        """
        try:
            result = self.sentiment_analyzer(text, truncation=True, max_length=512)[0]
            return {
                'label': result['label'],
                'score': result['score']
            }
        except Exception as e:
            logger.warning(f"Sentiment analysis failed for text: {text[:50]}... - {str(e)}")
            return {'label': 'UNKNOWN', 'score': 0.0}

class SentimentdNERModel(NERModel):
    def __init__(self, model_name: str, sentiment_model: str = None):
        super().__init__(model_name)
        self.sentiment_analyzer = None
        if sentiment_model:
            self.sentiment_analyzer = EntitySentimentAnalyzer(sentiment_model)

    def predict_with_sentiment(self, texts: List[str], batch_size: int = 8, 
                             context_window: int = 100) -> List[List[Dict]]:
        """
        Predict entities with sentiment analysis
        """
        if not self.sentiment_analyzer:
            raise ValueError("Sentiment analyzer not initialized")

        entities = self.predict(texts, batch_size)
        
        try:
            logger.info("Adding sentiment analysis to entities")
            return [
                self.sentiment_analyzer.analyze_entities(text, ents, context_window)
                for text, ents in zip(texts, entities)
            ]
        except Exception as e:
            logger.error(f"Failed to add sentiment analysis: {str(e)}")
            return entities