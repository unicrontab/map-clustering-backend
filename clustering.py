import json

try:
    from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans, estimate_bandwidth, MeanShift, AffinityPropagation, AgglomerativeClustering, DBSCAN, Birch
    from sklearn.neighbors import kneighbors_graph
    from sklearn.mixture import GaussianMixture

    import numpy as np
except:
    import unzip_requirements
    from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans, estimate_bandwidth, MeanShift, AffinityPropagation, AgglomerativeClustering, DBSCAN, Birch
    from sklearn.neighbors import kneighbors_graph
    from sklearn.mixture import GaussianMixture
    import numpy as np

def main(event, context):
    body = {}
    dataObject = []
    if 'body' in event.keys():
        body = json.loads(event['body'])
        dataObject = body['data']
    else:
        return createResponse({ 'error': 'no data'})
    
    X = np.empty(shape=[0,2])
    print(X)
    for address in dataObject:
        lat = float(address['location']['lat'])
        lng = float(address['location']['lng'])
        X = np.append(X, [[lat, lng]], axis=0)

    n_clusters = 2;
    if ('clusters' in body):
        n_clusters = body['clusters']

    algoName = 'kmeans'
    if ('algo' in body):
        algoName = body['algo']
        if algoName == 'spectral':
            algo = SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack', affinity='nearest_neighbors')
            algo.fit(X)
            prediction = algo.labels_.astype(np.int)
        if algoName == 'minikmeans':
            algo = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, n_init=10, max_no_improvement=10, verbose=0)
            algo.fit(X)
            prediction = algo.predict(X)
        if algoName == 'meanshift':
            bandwidth = estimate_bandwidth(X, quantile=0.3)
            algo = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            algo.fit(X)
            prediction = algo.labels_.astype(np.int)
        if algoName == 'affinity':
            algo = AffinityPropagation(damping=0.8)
            algo.fit(X)
            prediction = algo.labels_.astype(np.int)
        if algoName == 'agglo':
            connectivity = kneighbors_graph(X, n_neighbors=2, include_self=False)
            algo = AgglomerativeClustering(linkage='average', affinity='cityblock', n_clusters=n_clusters, connectivity=connectivity)
            algo.fit(X)
            prediction = algo.labels_.astype(np.int)
        if algoName == 'birch':
            algo = Birch(n_clusters=n_clusters, threshold=0.1, branching_factor=10)
            algo.fit(X)
            prediction = algo.labels_.astype(np.int)
        if algoName == 'gaussian':
            algo = GaussianMixture(n_components=n_clusters, covariance_type='full')
            algo.fit(X)
            prediction = algo.predict(X)
        else:
            algo = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
            algo.fit(X)
            prediction = algo.predict(X)
    else:
        algo = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
        prediction = algo.predict(X)




    print(prediction)

    clusterData = []
    for index, address in enumerate(dataObject):
        lat = float(address['location']['lat'])
        lng = float(address['location']['lng'])
        cluster = int(prediction[index])
        clusterData.append({
            'address': address['address'],
            'lat': lat,
            'lng': lng,
            'cluster': cluster,
        })
    
    dataWithMetaData = {
        'clusters': n_clusters,
        'algo': algoName,
        'addresses': len(dataObject),
        'clusterData': clusterData
    }
    return createResponse(dataWithMetaData)

def createResponse(modelOutput):
    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Credentials': 'true'
        },
        'body': json.dumps(modelOutput)
    }


if __name__ == '__main__':
    data = [
        {
            'address': '1 to -2',
            'location': {
                'lat': 1,
                'lng': -2,
            }
        },
        {
            'address': '2 to -2',
            'location': {
                'lat': 2,
                'lng': -2,
            }
        },
        {
            'address': '3 to -2',
            'location': {
                'lat': 3,
                'lng': -2,
            }
        },
        {
            'address': '10 to 3',
            'location': {
                'lat': 10,
                'lng': 3,
            }
        },
        {
            'address': '11 to 5',
            'location': {
                'lat': 11,
                'lng': 5,
            }
        },
        {
            'address': '9 to 4',
            'location': {
                'lat': 9,
                'lng': 4,
            }
        },
        {
            'address': '10 to 3',
            'location': {
                'lat': 15,
                'lng': 12,
            }
        },
        {
            'address': '11 to 5',
            'location': {
                'lat': 16,
                'lng': 13,
            }
        },
        {
            'address': '9 to 4',
            'location': {
                'lat': 17,
                'lng': 14,
            }
        },
        {
            'address': '9 to 4',
            'location': {
                'lat': 30,
                'lng': 14,
            }
        }
    ]
    body = {
        'algo': 'gaussian',
        'clusters': 3,
        'data': data
    }
    testEvent = {}
    testEvent['body'] = json.dumps(body)
    print(json.dumps(main(testEvent, ''), indent=4))
