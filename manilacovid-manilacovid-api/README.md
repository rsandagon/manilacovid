# Manila Covid API

> Modified on top of on mgorski's [Covid-19 Semantic Browser](https://github.com/gsarti/covid-papers-browser)

## Docker
* download the data `cd app` then run `/.download_data.sh`
* build the docker image with `/.build.sh`
* `docker run -d --name api -p 8000:80 rsandagon/manilacovid-api:latest`

## w/o Docker
* download the data `cd app` then run `/.download_data.sh`
* `pip install -r requirements.txt --no-cache-dir`
* `uvicorn main:app --reload`

## References
[http://arxiv.org/abs/1901.08149](http://arxiv.org/abs/1901.08149):

## Credits
* mgorski [Covid-19 Semantic Browser](https://github.com/gsarti/covid-papers-browser)