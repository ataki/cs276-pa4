package cs276.pa4;

import java.util.*;

import cs276.pa4.Util.IdfDictionary;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.functions.LinearRegression;

public class PointwiseLearner extends Learner {

    /**
     * Populates dataset with rows read from data.
     * Extracts labels by looking up corresponding
     * query / url value in labels map.
     * Uses data from idfs to calculate score.
     * Takes in optional indexMap to populate with indices
     */
    private void convertToRowsAndInsert(Instances dataset, Map<Query, List<Document>> data,
        Map<String, Map<String, Double>> labels, IdfDictionary idfs,
        Map<Query, Map<Document, Integer>> indexMap) {

        // Build data
        for (Query q : data.keySet()) {
            // query vector (idf scores)
            Map<String, Double> queryV = super.getQueryFreqs(q, idfs);
            for (Document d : data.get(q)) {
                double[] instance = new double[6];
                Map<String, Map<String, Double>> docTermFreqs = super.getDocTermFreqs(d, q);

                // term frequency vector for each field
                Map<String, Double> urlTFV = docTermFreqs.get("url");
                Map<String, Double> titleTFV = docTermFreqs.get("title");
                Map<String, Double> bodyTFV = docTermFreqs.get("body");
                Map<String, Double> headerTFV = docTermFreqs.get("header");
                Map<String, Double> anchorTFV = docTermFreqs.get("anchor");

                // construct instance vector of values
                // order is {url, title, body, header, anchor, relevance_score}
                instance[0] = super.multiplyQueryTermMappings(queryV, urlTFV);
                instance[1] = super.multiplyQueryTermMappings(queryV, titleTFV);
                instance[2] = super.multiplyQueryTermMappings(queryV, bodyTFV);
                instance[3] = super.multiplyQueryTermMappings(queryV, headerTFV);
                instance[4] = super.multiplyQueryTermMappings(queryV, anchorTFV);
                if (labels != null) instance[5] = labels.get(q.query).get(d.url);
                else instance[5] = 11; // for testing, this value is irrelevant

                // populate index mapping (for test functions)
                if (indexMap != null) {
                    int idx = dataset.numInstances();
                    if (!indexMap.containsKey(q))
                        indexMap.put(q, new HashMap<Document, Integer>());
                    Map<Document, Integer> mapping = indexMap.get(q);
                    if (!mapping.containsKey(d))
                        mapping.put(d, idx);
                }

                Instance inst = new DenseInstance(1.0, instance);
                dataset.add(inst);
            }
        }
    }

    ///////////////////// Public Methods ////////////////////////////

    @Override
    public Instances extract_train_features(String train_data_file,
            String train_rel_file, IdfDictionary idfs) {

        // Features
        Map<Query, List<Document>> trainData = null;

        // Labels
        Map<String, Map<String, Double>> relData = null;

        // Load training data
        try {
            trainData = Util.loadTrainData(train_data_file);
        }
        catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
        try {
            relData = Util.loadRelData(train_rel_file);
        }
        catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        // Dataset to build up and return
        Instances dataset = null;

        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        attributes.add(new Attribute("url_w"));
        attributes.add(new Attribute("title_w"));
        attributes.add(new Attribute("body_w"));
        attributes.add(new Attribute("header_w"));
        attributes.add(new Attribute("anchor_w"));
        attributes.add(new Attribute("relevance_score"));
        dataset = new Instances("train_dataset", attributes, 0);

        // Build data
        convertToRowsAndInsert(dataset, trainData, relData, idfs, null);

        /* Set last attribute as target */
        dataset.setClassIndex(dataset.numAttributes() - 1);

        return dataset;
    }

    @Override
    public Classifier training(Instances dataset) {
        LinearRegression model = new LinearRegression();
        try {
            model.buildClassifier(dataset);
        } catch (Exception e) {
            System.out.println("Unable to train data");
            e.printStackTrace();
            System.exit(1);
        }
        return model;
    }

    @Override
    public TestFeatures extract_test_features(String test_data_file,
            IdfDictionary idfs) {
        
        // Features
        Map<Query, List<Document>> testData = null;

        // Load test features
        try {
            testData = Util.loadTrainData(test_data_file);
        }
        catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        // Dataset to build up and return
        Instances dataset = null;

        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        attributes.add(new Attribute("url_w"));
        attributes.add(new Attribute("title_w"));
        attributes.add(new Attribute("body_w"));
        attributes.add(new Attribute("header_w"));
        attributes.add(new Attribute("anchor_w"));
        attributes.add(new Attribute("relevance_score"));
        dataset = new Instances("test_dataset", attributes, 0);

        // tracks map of (Query -> (Document, Index)) in our data
        Map<Query, Map<Document, Integer>> indexMap = new HashMap<Query, Map<Document, Integer>>();

        // Build data
        convertToRowsAndInsert(dataset, testData, null, idfs, indexMap);

        /* Set last attribute as target */
        dataset.setClassIndex(dataset.numAttributes() - 1);

        TestFeatures tFeatures = new TestFeatures();
        tFeatures.features = dataset;
        tFeatures.index_map = indexMap;

        return tFeatures;
    }

    @SuppressWarnings("MismatchedQueryAndUpdateOfCollection")
    @Override
    public Map<String, List<String>> testing(TestFeatures tf,
            Classifier model) {

        Instances dataSet = tf.features;
        Map<String, List<String>> results = new HashMap<String, List<String>>() ;

        for (Query q : tf.index_map.keySet()) {
            // use sorted map to avoid writing a comparator
            SortedMap<Double, String> resultForQuery = new TreeMap<Double, String>();
            for (Document d : tf.index_map.get(q).keySet()) {
                Integer idx = tf.index_map.get(q).get(d);
                Instance row = dataSet.get(idx);
                double score;
                try {
                    // negative so we can sort by descending automatically
                    score = -model.classifyInstance(row);
                    resultForQuery.put(score, d.url);
                } catch (Exception e) {
                    System.out.println("Error occurred while classifying");
                    e.printStackTrace();
                    System.exit(1);
                }
            }
            ArrayList<String> sortedResultsForQuery = new ArrayList<String>();
            for (Map.Entry<Double, String> entry : resultForQuery.entrySet()) {
                sortedResultsForQuery.add(entry.getValue());
            }
            results.put(q.query, sortedResultsForQuery);
        }
        return results;
    }

}
