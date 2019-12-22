.PHONY: all

compile:
	sbt compile

build:
	sbt assembly
	mkdir -p build/
	cp target/scala-2.11/tweets-transformer-with-deps-0.2.0.jar build/tweets-transformer.jar
	cp bin/tweets-transformer.sh build/tweets-transformer.sh

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
	tar -zcvf dist/tweets-transformer-0.2.0.tgz build/

all: clean build dist
