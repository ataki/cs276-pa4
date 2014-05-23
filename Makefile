task1:
	@echo "Running linear regression, pointwise."
	@echo "The target for the NDCG score is 0.85"
	@echo "-----------------------------------------"
	./run.sh pa4.signal.train pa4.rel.train pa4.signal.dev pa4.rel.dev 1

task2:
	@echo "Running svm, pairwise. Set kernel type inside code"
	@echo "The target for the NDCG score is 0.85 for linear, 0.87 for non-linear"
	@echo "----------------------------------------------------------------------"
		./run.sh pa4.signal.train pa4.rel.train pa4.signal.dev pa4.rel.dev 2

task3:
	@echo "Running custom features with svm, pairwise."
	@echo "The target for the NDCG score is >0.85 / >0.87"
	@echo "-----------------------------------------------"
	./run.sh pa4.signal.train pa4.rel.train pa4.signal.dev pa4.rel.dev 2


