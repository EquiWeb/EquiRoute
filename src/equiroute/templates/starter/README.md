# EquiRoute starter router

This project contains a complete three-route parent router and an `add_shipping_address` continuation. All paths in the configuration files are relative to the `config/` directory.

All model dependencies are declared by the project, so the workflow needs no manual package changes.

Follow the complete acceptance workflow in order:

```console
$ uv sync
$ uv run equiroute validate config/parent.yaml
$ uv run equiroute validate config/add-shipping.yaml
$ uv run equiroute train config/parent.yaml
$ uv run equiroute evaluate artifacts/parent-router --data data/parent/test.jsonl
$ uv run python -c 'from transformers import AutoModelForCausalLM, AutoTokenizer; path = "artifacts/parent-router/model"; tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True); model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True)'
$ uv run equiroute continue --from artifacts/parent-router --config config/add-shipping.yaml
$ uv run equiroute evaluate artifacts/add-shipping-router --data data/add-shipping/test.jsonl
```

The Python command independently loads the parent export with standard Transformers APIs and does not contact the network.

Optionally verify the completed child export:

```console
$ uv run equiroute export artifacts/add-shipping-router
```

The child registry retains the parent routes as an exact ordered prefix. Its train, validation, and test partitions replay each parent route and add the shipping-address route. Its separate regression dataset is sealed old-route evidence and must not overlap those child partitions.
