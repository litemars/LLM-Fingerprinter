"""Shared identification policy, independent of transport and artifact paths."""


class IdentificationPipeline:
    """Use full-fingerprint family templates for identity and novelty rejection.

    The ensemble remains diagnostic evidence. An uncertain or missing family
    template cannot be overridden by a closed-set prediction or a model-name
    match. Templates are injected or taken from the trained classifier; this
    class never loads unrelated artifacts from the current environment.
    """

    def __init__(self, classifier=None, family_templates=None, model_templates=None):
        self.classifier = classifier
        self.family_templates = (family_templates if family_templates is not None
                                 else getattr(classifier, "family_templates", None))
        self.model_templates = model_templates

    def classify(self, vector, incomplete=False):
        ensemble = {}
        if self.classifier is not None and self.classifier.is_trained:
            family, confidence, probabilities, ood = (
                self.classifier.predict_with_confidence(vector))
            ensemble = {
                "predicted_family": family,
                "confidence": float(confidence),
                "all_probabilities": probabilities,
                "ood_detected": bool(ood.get("is_ood", False)),
                "ood_details": ood,
            }

        result = {
            "family": "unknown",
            "predicted_family": ensemble.get("predicted_family") or "unknown",
            "confidence": 0.0,
            "confidence_kind": "unavailable",
            "all_probabilities": ensemble.get("all_probabilities", {}),
            "ensemble_result": ensemble,
            "ood_detected": True,
            "ood_details": {},
            "decision_policy": "family_templates_v1",
            "family_source": "family_template",
        }
        if incomplete:
            result["decision_reason"] = "partial_fingerprint"
            result["templates_skipped_reason"] = "early_stopped"
            return result
        if self.family_templates is None or not self.family_templates.is_built:
            result["decision_reason"] = "family_templates_unavailable"
            return result

        try:
            template = self.family_templates.classify(vector)
        except Exception as error:
            result["error"] = f"Family template classification failed: {error}"
            result["decision_reason"] = "invalid_fingerprint_or_templates"
            return result

        accepted = not template["is_ood"]
        result.update({
            "family": template["predicted_family"] if accepted else "unknown",
            "predicted_family": template["predicted_family"],
            "confidence": template["confidence"],
            "confidence_kind": "cosine_similarity",
            "ood_detected": not accepted,
            "ood_details": {"reason": template.get("ood_reason"),
                            "distance": template["distance"]},
            "decision_reason": "accepted" if accepted else template.get("ood_reason", "unknown"),
            "template_result": template,
            "ensemble_disagrees": bool(ensemble.get("predicted_family") and
                                        ensemble["predicted_family"] != template["predicted_family"]),
        })

        if self.model_templates is not None and self.model_templates.is_built:
            try:
                model = self.model_templates.classify(vector, top_k=4)
                # A version match is valid only inside an accepted family.
                model_accepted = (accepted and not model["is_ood"] and
                                  model.get("inferred_family") == result["family"])
                result["model_estimate"] = {
                    "model": model["predicted_family"] if model_accepted else "unknown",
                    "predicted_model": model["predicted_family"],
                    "confidence": model["confidence"],
                    "confidence_kind": "cosine_similarity",
                    "distance": model["distance"],
                    "is_ood": not model_accepted,
                    "ood_reason": (model.get("ood_reason") if model["is_ood"] else
                                   None if model_accepted else "family_not_confirmed"),
                    "ranked": model["ranked"],
                    "inferred_family": model.get("inferred_family"),
                }
            except Exception as error:
                # Version estimates are optional; do not discard a valid family.
                result["model_estimate_error"] = str(error)
        return result
