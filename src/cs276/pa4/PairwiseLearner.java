package cs276.pa4;

import java.util.*;
import cs276.pa4.Util.IdfDictionary;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.classifiers.functions.LibSVM;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;
import weka.core.SelectedTag;

public class PairwiseLearner extends Learner {
    private LibSVM model;

    public PairwiseLearner(boolean isLinearKernel){
        try{
            model = new LibSVM();
        } catch (Exception e){
            e.printStackTrace();
        }

        if(isLinearKernel){
            model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
        }
    }
  
    public PairwiseLearner(double C, double gamma, boolean isLinearKernel){
        try{
            model = new LibSVM();
        } catch (Exception e){
            e.printStackTrace();
        }

        model.setCost(C);
        model.setGamma(gamma); // only matter for RBF kernel
        if(isLinearKernel){
            model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
        }
    }
  
    /* Instance is a length-12 array of doubles; the first 5 entries
     * represent features of x1 and the latter 6 represent features of x2. 
     * The last field of "instance" will be filled with the classification
     */
    private Instance constructInstanceFromQueryDocPair(double[] instance, double x1Score, double x2Score) {
        assert instance.length == 11;
        int classif = x1Score > x2Score ? 1 : -1;
        instance[10] = (double)classif;
        return new DenseInstance(1.0, instance);
    }

    /**
     * convertToRowsAndInsert.
     * Populates dataset with pairs of rows read from data.
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
            List<Document> documentList = data.get(q);

            for (int i = 0; i < documentList.size(); i++) {
                double[] instance = new double[11];

                // compute first doc (x1) values

                Document d1 = documentList.get(i);
                Map<String, Map<String, Double>> d1TFs = super.getDocTermFreqs(d1, q);

                // term frequency vector for each field
                Map<String, Double> d1UrlTFV = d1TFs.get("url");
                Map<String, Double> d1TitleTFV = d1TFs.get("title");
                Map<String, Double> d1BodyTFV = d1TFs.get("body");
                Map<String, Double> d1HeaderTFV = d1TFs.get("header");
                Map<String, Double> d1AnchorTFV = d1TFs.get("anchor");

                // order is {url, title, body, header, anchor, relevance_class}
                instance[0] = super.multiplyQueryTermMappings(queryV, d1UrlTFV);
                instance[1] = super.multiplyQueryTermMappings(queryV, d1TitleTFV);
                instance[2] = super.multiplyQueryTermMappings(queryV, d1BodyTFV);
                instance[3] = super.multiplyQueryTermMappings(queryV, d1HeaderTFV);
                instance[4] = super.multiplyQueryTermMappings(queryV, d1AnchorTFV);

                for (int j = 0; j < i; j++) {
                    Document d2 = documentList.get(j);
                    Map<String, Map<String, Double>> d2TFs = super.getDocTermFreqs(d2, q);

                    // term frequency vector for each field
                    Map<String, Double> d2UrlTFV = d2TFs.get("url");
                    Map<String, Double> d2TitleTFV = d2TFs.get("title");
                    Map<String, Double> d2BodyTFV = d2TFs.get("body");
                    Map<String, Double> d2HeaderTFV = d2TFs.get("header");
                    Map<String, Double> d2AnchorTFV = d2TFs.get("anchor");

                    instance[5] = super.multiplyQueryTermMappings(queryV, d2UrlTFV);
                    instance[6] = super.multiplyQueryTermMappings(queryV, d2TitleTFV);
                    instance[7] = super.multiplyQueryTermMappings(queryV, d2BodyTFV);
                    instance[8] = super.multiplyQueryTermMappings(queryV, d2HeaderTFV);
                    instance[9] = super.multiplyQueryTermMappings(queryV, d2AnchorTFV);

                    double score1 = labels.get(q.query).get(d1.url);
                    double score2 = labels.get(q.query).get(d2.url);
                    dataset.add(constructInstanceFromQueryDocPair(instance, score1, score2));
                }

                // populate index mapping (for test functions)
                if (indexMap != null) {
                    int idx = dataset.numInstances();
                    if (!indexMap.containsKey(q))
                        indexMap.put(q, new HashMap<Document, Integer>());
                    Map<Document, Integer> mapping = indexMap.get(q);
                    if (!mapping.containsKey(d1))
                        mapping.put(d1, idx);
                }
            }
        }
    }

    /* Performs standardization using weka */
    private Instances scaleDataset(Instances dataset) throws Exception {
        Standardize filter = new Standardize();
        filter.setInputFormat(dataset);
        return Filter.useFilter(dataset, filter);
    }

	@Override
	public Instances extract_train_features(String train_data_file,
			String train_rel_file, IdfDictionary idfs) {

        // Features
        Map<Query, List<Document>> trainData = null;

        // Labels
        Map<String, Map<String, Double>> relData = null;

        try {
            trainData = Util.loadTrainData(train_data_file);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        try {
            relData = Util.loadRelData(train_rel_file);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        // Dataset to build up and return
        Instances dataset = null;

        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        // x1
        attributes.add(new Attribute("url_w_1"));
        attributes.add(new Attribute("title_w_1"));
        attributes.add(new Attribute("body_w_1"));
        attributes.add(new Attribute("header_w_1"));
        attributes.add(new Attribute("anchor_w_1"));
        // x2
        attributes.add(new Attribute("url_w_2"));
        attributes.add(new Attribute("title_w_2"));
        attributes.add(new Attribute("body_w_2"));
        attributes.add(new Attribute("header_w_2"));
        attributes.add(new Attribute("anchor_w_2"));
        // label (+1 or -1)
        attributes.add(new Attribute("relevance_class"));
        dataset = new Instances("train_dataset", attributes, 0);

        // Build data
        convertToRowsAndInsert(dataset, trainData, relData, idfs, null);

        // Set last attribute as target
        dataset.setClassIndex(dataset.numAttributes() - 1);

        // Standardize. Note: This is specific to SVM Pairwise
        try {
            dataset = scaleDataset(dataset);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        return dataset;
    }

	@Override
	public Classifier training(Instances dataset) {
        try {
            this.model.buildClassifier(dataset);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
		return this.model;
	}

    /* extract_test_features is basically same as PointWise */
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
        // x1
        attributes.add(new Attribute("url_w_1"));
        attributes.add(new Attribute("title_w_1"));
        attributes.add(new Attribute("body_w_1"));
        attributes.add(new Attribute("header_w_1"));
        attributes.add(new Attribute("anchor_w_1"));
        // x2
        attributes.add(new Attribute("url_w_2"));
        attributes.add(new Attribute("title_w_2"));
        attributes.add(new Attribute("body_w_2"));
        attributes.add(new Attribute("header_w_2"));
        attributes.add(new Attribute("anchor_w_2"));
        // label (+1 or -1)
        attributes.add(new Attribute("relevance_class"));
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

	@Override
	public Map<String, List<String>> testing(TestFeatures tf,
			Classifier model) {
		/*
		 * @TODO: Your code here
         * Do Pairwise comparison here
		 */
		return null;
	}

}
