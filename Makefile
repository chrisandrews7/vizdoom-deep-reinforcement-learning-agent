.PHONY: agentic-train
agentic-train: guard-ENV
	uv run --env-file $(ENV) agentic_train.py 

.PHONY: train
train: guard-ENV
	uv run --env-file $(ENV) train.py 

guard-%:
	@ if [ "${${*}}" = "" ]; then \
			echo "Environment variable $* not set"; \
			exit 1; \
	fi
