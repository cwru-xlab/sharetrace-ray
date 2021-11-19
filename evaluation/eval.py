import logging
from logging import config

from evaluation import synthetic
from sharetrace import propagation, search, util

logging.config.dictConfig(util.logging_config())
search_logger = logging.getLogger('contact-search')
prop_logger = logging.getLogger('propagation-ray')


def main():
    n = 1000
    histories = synthetic.to_geohashes(synthetic.load_histories(n))
    scores = synthetic.load_scores(n)
    contact_search = search.KdTreeContactSearch(
        search_logger, min_dur=900, workers=-1)
    contacts = contact_search.search(histories)
    risk_prop = propagation.RayRiskPropagation(
        prop_logger, parts=4, early_stop=10_000, tol=0.3, timeout=3)
    risk_prop.setup(scores, contacts)
    risk_prop.run()


if __name__ == '__main__':
    main()
