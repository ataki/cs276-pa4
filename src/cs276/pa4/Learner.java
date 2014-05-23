package cs276.pa4;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import cs276.pa4.Util.IdfDictionary;

import weka.classifiers.Classifier;
import weka.core.Instances;

public abstract class Learner {

	/* Construct training features matrix */
	public abstract Instances extract_train_features(String train_data_file, String train_rel_file, IdfDictionary idfs);

	/* Train the model */
	public abstract Classifier training (Instances dataset);
	
	/* Construct testing features matrix */
	public abstract TestFeatures extract_test_features(String test_data_file, IdfDictionary idfs);
	
	/* Test the model, return ranked queries */
	public abstract Map<String, List<String>> testing(TestFeatures tf, Classifier model);

    /////////// Methods shared by Linear / SVM Learners ///////////////

    /* does q^T * t_f */
    protected double multiplyQueryTermMappings(Map<String, Double> query,
         Map<String, Double> termFreqs) {

        double sum = 0;
        for (String term : query.keySet()) {
            sum += query.get(term) * termFreqs.get(term);
        }
        return sum;
    }

    /////////////////////// Begin PA3 code //////////////////////////////

    String[] TFTYPES = {"url", "title", "body", "header", "anchor"};

    private String[] splitUrl(String url) {
        return url.split("\\W+");
    }

    private String[] splitTitleOrHeader(String sentence) {
        return sentence.split("\\s+");
    }

    private Map<String, Double> createFreqMap(String[] terms) {
        Map<String, Double> freqMap = new HashMap<String, Double>();
        for (String term : terms) {
            if (!freqMap.containsKey(term)) {
                freqMap.put(term, 1.0);
            } else {
                freqMap.put(term, freqMap.get(term) + 1.0);
            }
        }
        return freqMap;
    }

    /*
     * Creates mapping of idf scores for each term in a query.
     * Handles duplicates by summing their idf weights
     */
    public Map<String, Double> getQueryFreqs(Query q, IdfDictionary idfs) {
        Map<String, Double> tfQuery = new HashMap<String, Double>();

        for (String word : q.words) {
            word = word.toLowerCase();

            double weight = 0.0;
            if (tfQuery.containsKey(word)) {
                weight = tfQuery.get(word);
            }

            // apply idf weights
            weight += idfs.getTermFreq(word); // default smoothing
            tfQuery.put(word, weight);
        }

        return tfQuery;
    }

    /*
     * Creates the document frequencies for each field.
     * Returns a mapping of field -> (term -> raw_term_frequencies)
     * Duplicate terms in the query are handled cumulatively (see note)
     */
    public Map<String, Map<String, Double>> getDocTermFreqs(Document d, Query q) {
        //map from tf type -> queryWord -> score
        Map<String, Map<String, Double>> tfs = new HashMap<String, Map<String, Double>>();

        // when calculating raw scores, lower-case terms of all fields
        String docUrl = d.url.toLowerCase();
        String docTitle = d.title.toLowerCase();
        List<String> docHeaders = new ArrayList<String>();
        if (d.headers != null) {
            for (String s : d.headers) {
                docHeaders.add(s.toLowerCase());
            }
        }
        Map<String, List<Integer>> docBodyHits = new HashMap<String, List<Integer>>();
        if (d.body_hits != null) {
            for (String term : d.body_hits.keySet()) {
                docBodyHits.put(term.toLowerCase(), d.body_hits.get(term));
            }
        }
        Map<String, Integer> docAnchors = new HashMap<String, Integer>();
        if (d.anchors != null) {
            for (String term : d.anchors.keySet()) {
                docAnchors.put(term.toLowerCase(), d.anchors.get(term));
            }

        }
        List<String> queryWords = new ArrayList<String>();
        for (String word : q.words) {
            queryWords.add(word.toLowerCase());
        }

        // url
        Map<String, Double> urlMap = createFreqMap(splitUrl(docUrl));

        // title
        Map<String, Double> titleMap = createFreqMap(splitTitleOrHeader(docTitle));

        // headers
        Map<String, Double> headersMap = new HashMap<String, Double>();
        for (String header : docHeaders) {
            String headerTerms[] = splitTitleOrHeader(header);
            for (String term : headerTerms) {
                if (!headersMap.containsKey(term)) {
                    headersMap.put(term, 1.0);
                } else {
                    headersMap.put(term, headersMap.get(term) + 1.0);
                }
            }
        }

        // body_hits
        Map<String, Double> bodyMap = new HashMap<String, Double>();
        for (String term : docBodyHits.keySet()) {
            double count = 0.0;
            if (bodyMap.containsKey(term))
                count = bodyMap.get(term);
            bodyMap.put(term, count + (double) docBodyHits.get(term).size());
        }

        // anchors
        Map<String, Double> anchorsMap = new HashMap<String, Double>();
        for (String anchor : docAnchors.keySet()) {
            String anchorTerms[] = anchor.split("\\s+");
            for (String term : anchorTerms) {
                double termCount = (double) docAnchors.get(anchor);
                if (!anchorsMap.containsKey(term)) {
                    anchorsMap.put(term, termCount);
                } else {
                    anchorsMap.put(term, anchorsMap.get(term) + termCount);
                }
            }
        }

        Map<String, Map<String, Double>> allMaps = new HashMap<String, Map<String, Double>>();

        allMaps.put("url", urlMap);
        allMaps.put("title", titleMap);
        allMaps.put("header", headersMap);
        allMaps.put("body", bodyMap);
        allMaps.put("anchor", anchorsMap);

        for (String type : TFTYPES) {
            Map<String, Double> rawCountMap = new HashMap<String, Double>();
            Map<String, Double> typeCountMap = allMaps.get(type);
            for (String queryWord : queryWords) {
                double count = 0.0;
                if (typeCountMap.containsKey(queryWord))
                    count = typeCountMap.get(queryWord);

                // handle duplicate words in query
                // by accumulating scores e.g.
                // query "hello world hello" is treated
                // differently from "hello world"

                if (!rawCountMap.containsKey(queryWord))
                    rawCountMap.put(queryWord, count);
                else
                    rawCountMap.put(queryWord, rawCountMap.get(queryWord) + count);
            }
            tfs.put(type, rawCountMap);
        }
        return tfs;
    }

    /////////////////////// End PA3 Code //////////////////////////////////


}

