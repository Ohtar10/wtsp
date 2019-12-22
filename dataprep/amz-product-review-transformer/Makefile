.PHONY: all

compile:
	sbt compile

build:
	sbt assembly
	mkdir -p build/
	cp target/scala-2.11/amz-prod-review-transformer-with-deps-0.1.0.jar build/product-doc-transformer.jar
	cp bin/product-doc-transformer.sh build/product-doc-transformer.sh
	cp src/main/resources/mapping/category_mappings.yml build/category_mappings.yml

clean: clean-build clean-dist

clean-build:
	sbt clean
	rm -rf target/
	rm -rf build/
	rm -rf spark-warehouse/

clean-dist:
	rm -rf dist/

test:
	sbt test

dist:
	mkdir -p dist/
	tar -zcvf dist/product-doc-transformer-0.1.0.tgz build/

all: clean build dist
