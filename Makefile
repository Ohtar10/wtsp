.PHONY: all

clean-dataprep:
	(cd dataprep/amz-product-review-transformer; make clean)
	(cd dataprep/tweets-transformer; make clean)

clean-cli:
	(cd cli; make clean)

clean: clean-dataprep clean-cli

install-dataprep:
	(cd dataprep/amz-product-review-transformer; make all)
	(cd dataprep/tweets-transformer; make all)

install-cli:
	(cd cli; make all)

install: install-dataprep install-cli

all: clean install