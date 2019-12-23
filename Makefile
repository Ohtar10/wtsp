.PHONY: all

clean-dataprep:
	(cd dataprep/amz-product-review-transformer; make clean)
	(cd dataprep/tweets-transformer; make clean)

clean-cli:
	(cd cli; make clean)

clean: clean-dataprep clean-cli clean-docker

clean-docker:
	rm -rf docker-build

install-dataprep:
	(cd dataprep/amz-product-review-transformer; make all)
	(cd dataprep/tweets-transformer; make all)

install-cli:
	(cd cli; make all)

install: install-dataprep install-cli

docker-clean:
	rm -rf docker-build
	make clean-cli

docker-deps:
	(cd cli; make dist)

docker:
	mkdir -p docker-build
	cp env/docker/Dockerfile docker-build/Dockerfile
	cp env/conda/wtsp-full-linux-no-gpu.yaml docker-build/environment.yaml
	cp cli/dist/wtsp-0.1.0.tar.gz docker-build/wtsp-0.1.0.tar.gz
	(cd docker-build; docker image build -t ohtar10/wtsp:0.1.0 .)

docker-all: docker-clean docker-deps docker

all: clean install