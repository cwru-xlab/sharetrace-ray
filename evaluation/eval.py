from evaluation import synthetic
from sharetrace import propagation, search


def main():
    n = 1000
    histories = synthetic.to_geohashes(synthetic.load_histories(n))
    scores = synthetic.load_scores(n)
    contact_search = search.KdTreeContactSearch(min_dur=900, workers=-1)
    contacts = contact_search.search(histories)
    risk_prop = propagation.RayRiskPropagation(
        parts=4, early_stop=10_000, tol=0.3, timeout=3)
    risk_prop.setup(scores, contacts)
    risk_prop.run()


if __name__ == '__main__':
    main()
