package cs276.pa4;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Util {

  public static class IdfDictionary {
    private Map<String, Double> dfs;
    private double totalDocCount;

    public IdfDictionary(Map<String, Double> dfs, double totalDocCount) {
        this.dfs = dfs;
        this.totalDocCount = totalDocCount;
    }

    // by default, offer normalization
    public double getTermFreq(String term) {
        return getTermFreq(term, true);
    }

    public double getTermFreq(String term, boolean useSmoothing) {
        double freq = 0.0;
        if (dfs.containsKey(term))
            freq = dfs.get(term);
        else
            if (useSmoothing)
                freq = Math.log(totalDocCount + 1);
        return freq;
    }

  }

  public static Map<Query,List<Document>> loadTrainData (String feature_file_name) throws Exception {
    Map<Query, List<Document>> result = new HashMap<Query, List<Document>>();

    File feature_file = new File(feature_file_name);
    if (!feature_file.exists() ) {
      System.err.println("Invalid feature file name: " + feature_file_name);
      return null;
    }

    BufferedReader reader = new BufferedReader(new FileReader(feature_file));
    String line = null, anchor_text = null;
    Query query = null;
    Document doc = null;
    int numQuery=0; int numDoc=0;
    while ((line = reader.readLine()) != null) {
      String[] tokens = line.split(":", 2);
      String key = tokens[0].trim();
      String value = tokens[1].trim();

      if (key.equals("query")){
        query = new Query(value);
        numQuery++;
        result.put(query, new ArrayList<Document>());
      } else if (key.equals("url")) {
        doc = new Document();
        doc.url = new String(value);
        result.get(query).add(doc);
        numDoc++;
      } else if (key.equals("title")) {
        doc.title = new String(value);
      } else if (key.equals("header"))
      {
        if (doc.headers == null)
          doc.headers =  new ArrayList<String>();
        doc.headers.add(value);
      } else if (key.equals("body_hits")) {
        if (doc.body_hits == null)
          doc.body_hits = new HashMap<String, List<Integer>>();
        String[] temp = value.split(" ", 2);
        String term = temp[0].trim();
        List<Integer> positions_int;

        if (!doc.body_hits.containsKey(term))
        {
          positions_int = new ArrayList<Integer>();
          doc.body_hits.put(term, positions_int);
        } else
          positions_int = doc.body_hits.get(term);

        String[] positions = temp[1].trim().split(" ");
        for (String position : positions)
          positions_int.add(Integer.parseInt(position));

      } else if (key.equals("body_length"))
        doc.body_length = Integer.parseInt(value);
      else if (key.equals("pagerank"))
        doc.page_rank = Integer.parseInt(value);
      else if (key.equals("anchor_text")) {
        anchor_text = value;
        if (doc.anchors == null)
          doc.anchors = new HashMap<String, Integer>();
      }
      else if (key.equals("stanford_anchor_count"))
        doc.anchors.put(anchor_text, Integer.parseInt(value));      
    }

    reader.close();
    System.err.println("# Signal file " + feature_file_name + ": number of queries=" + numQuery + ", number of documents=" + numDoc);

    return result;
  }

  public static IdfDictionary loadDFs(String dfFile) throws IOException {
    Map<String,Double> dfs = new HashMap<String, Double>();
    double totalDocumentCount = 0;

    BufferedReader br = new BufferedReader(new FileReader(dfFile));
    String line;
    while((line=br.readLine())!=null){
      line = line.trim();
      if(line.equals("")) continue;
      String[] tokens = line.split("\\s+");
      dfs.put(tokens[0], Double.parseDouble(tokens[1]));
      totalDocumentCount += 1.0;
    }
    br.close();

    // TODO How to remove this hack?
    // The way we're given the DF's gives us no info about
    // how many total docs there are. On average, we estimate
    // that the total document count is (# unique terms) / C
    // Based on experimentation, C=5.5 seems to be the most
    // accurate, yielding the highest score.

    totalDocumentCount /= 5.5;

    // create idf from dfs
    Map<String, Double> idfs = new HashMap<String, Double>(dfs.keySet().size());
    for (String term : dfs.keySet()) {
        double freq = dfs.get(term);
        double idf = Math.log((totalDocumentCount + 1.0) / (freq + 1.0));
        idfs.put(term, idf);
    }

    return new IdfDictionary(idfs, totalDocumentCount);
  }

  /* query -> (url -> score) */
  public static Map<String, Map<String, Double>> loadRelData(String rel_file_name) throws IOException{
    Map<String, Map<String, Double>> result = new HashMap<String, Map<String, Double>>();

    File rel_file = new File(rel_file_name);
    if (!rel_file.exists() ) {
      System.err.println("Invalid feature file name: " + rel_file_name);
      return null;
    }

    BufferedReader reader = new BufferedReader(new FileReader(rel_file));
    String line = null, query = null, url = null;
    int numQuery=0; 
    int numDoc=0;
    while ((line = reader.readLine()) != null) {
      String[] tokens = line.split(":", 2);
      String key = tokens[0].trim();
      String value = tokens[1].trim();

      if (key.equals("query")){
        query = value;
        result.put(query, new HashMap<String, Double>());
        numQuery++;
      } else if (key.equals("url")){
        String[] tmps = value.split(" ", 2);
        url = tmps[0].trim();
        double score = Double.parseDouble(tmps[1].trim());
        result.get(query).put(url, score);
        numDoc++;
      }
    }	
    reader.close();
    System.err.println("# Rel file " + rel_file_name + ": number of queries=" + numQuery + ", number of documents=" + numDoc);
    
    return result;
  }

  public static void main(String[] args) {
    try {
      System.out.print(loadRelData(args[0]));
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}
