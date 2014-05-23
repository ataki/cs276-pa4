package cs276.pa4;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import cs276.pa4.Util.*;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.functions.LinearRegression;

public class PointwiseLearner extends Learner {


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
        // TODO Implement sub-linear scaling
        for (Query q : trainData.keySet()) {
            Map<String, Double> queryV = getQueryFreqs(q, idfs);
            for (Document d : trainData.get(q)) {
                double[] instance = new double[5];
                Map<String, Map<String, Double>> docTermFreqs = getDocTermFreqs(d, q);

                // order is {url, title, body, header, anchor, relevance_score}
                Map<String, Double> url_tfs = docTermFreqs.get("url");
                Map<String, Double> title_tfs = docTermFreqs.get("title");
                Map<String, Double> body_tfs = docTermFreqs.get("body");
                Map<String, Double> header_tfs = docTermFreqs.get("header");
                Map<String, Double> anchor_tfs = docTermFreqs.get("anchor");

                instance[0] = multiplyQueryTermMappings(queryV, url_tfs);
                instance[1] = multiplyQueryTermMappings(queryV, title_tfs);
                instance[2] = multiplyQueryTermMappings(queryV, body_tfs);
                instance[3] = multiplyQueryTermMappings(queryV, header_tfs);
                instance[4] = multiplyQueryTermMappings(queryV, anchor_tfs);
                instance[5] = getRelevanceScore(q, d, docTermFreqs, idfs);

                Instance inst = new DenseInstance(1.0, instance);
                dataset.add(inst);
            }
        }

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
        /*
         * @TODO: Your code here
         */
        return null;
    }

    @Override
    public Map<String, List<String>> testing(TestFeatures tf,
            Classifier model) {
        /*
         * @TODO: Your code here
         */
        return null;
    }

}
