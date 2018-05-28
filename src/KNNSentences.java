import java.io.*;
import java.nio.file.FileSystems;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.*;
import weka.core.stemmers.LovinsStemmer;
import weka.core.tokenizers.WordTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class KNNSentences {

	public Instances createDataset(String directoryPath) throws Exception {

		String filePathSW = FileSystems.getDefault().getPath("src/word_lists/stopwords.txt").toAbsolutePath().toString();
		BufferedReader bReaderSW = null;
		List<String> swordArrayList = new ArrayList<String>();
		List<String> uniqueWordList = new ArrayList<String>();
		bReaderSW = new BufferedReader(new FileReader(filePathSW));
		String SWLine =null;
		while ((SWLine = bReaderSW.readLine() ) != null) 
		{       
			swordArrayList.add(SWLine);            
		}
		bReaderSW.close();


		ArrayList<Attribute> atts = new ArrayList<Attribute>();
		ArrayList<String> sentenceLabels = new ArrayList<String>(5); 
		sentenceLabels.add("AIMX"); 
		sentenceLabels.add("OWNX"); 
		sentenceLabels.add("CONT"); 
		sentenceLabels.add("BASE"); 
		sentenceLabels.add("MISC");
		atts.add(new Attribute("class", true));
		atts.add(new Attribute("sentence", true));
		Instances instancesData = new Instances("text_files_in_" + directoryPath, atts, 0);

		File dir = new File(directoryPath);
		String[] files = dir.list();
		Arrays.sort(files);
		Map<String, List<String>> mapLabels = new HashMap<>();
		Map<String, String> mapSentences = new HashMap<>();
		List<String> labels = Arrays.asList("AIMX", "OWNX", "CONT", "BASE", "MISC");  

		for (int i = 0; i < files.length; i++) {
			if (files[i].endsWith(".txt")) {
				try {
					double[] newInst = new double[2];
					//newInst[0] = (double)data.attribute(0).addStringValue(files[i]);

					int index = files[i].lastIndexOf("_");
					String fileName = files[i].substring(0, index).trim();

					File txt = new File(directoryPath + File.separator + files[i]);
					BufferedReader br = new BufferedReader(new FileReader(txt));

					int sentenceCount = 1;
					String lineData = null;
					while ((lineData = br.readLine()) != null) {
						boolean expertCheck = (lineData.contains("\t")) ? true : false;
						String[] splitClass = (expertCheck) ? lineData.split("\t") : lineData.split(" ");

						if(labels.contains(splitClass[0])) {
							String labelData = splitClass[0].trim();
							String sentenceData = (expertCheck) ? splitClass[1].trim() : lineData.replace(labelData, "").trim();


							String sentenceClass = fileName + "_S" + sentenceCount;
							String sentenceContent = fileName + "_S" + sentenceCount + "_content";

							List<String> sentenceClassData = (mapLabels.get(sentenceClass) == null) ? new ArrayList<String>() : mapLabels.get(sentenceClass);
							sentenceClassData.add(labelData);
							mapLabels.put(sentenceClass, sentenceClassData);
							mapSentences.put(sentenceContent, sentenceData);

							WordTokenizer wordTokenizer = new WordTokenizer();
							wordTokenizer.tokenize(sentenceData.replaceAll("[^a-zA-Z\\s]", ""));
							while(wordTokenizer.hasMoreElements()) 
							{
								String word = wordTokenizer.nextElement();
								String wordCompare = word.toLowerCase();
								if(!swordArrayList.contains(wordCompare) && !wordCompare.contains("@") && !wordCompare.contains("#") )
								{
									if(!uniqueWordList.contains(word.toLowerCase()))
									{
										uniqueWordList.add(word.toLowerCase());
									}
								}
							}

							sentenceCount++;
						}

					}
					br.close();
				} catch (Exception e) {
					//System.err.println("failed to convert file: " + directoryPath + File.separator + files[i]);
				}
			}
		}

		File arff_file = new File(FileSystems.getDefault().getPath("src/sentenceAnalysis.arff").toAbsolutePath().toString());
		PrintWriter arff_writer = new PrintWriter(arff_file.toString(), "UTF-8");
		arff_writer.println("@relation health_care");
		arff_writer.println();
		for(int i=0 ; i< uniqueWordList.size() ; i++)
		{
			arff_writer.println("@attribute "+uniqueWordList.get(i)+" numeric");
		}
		arff_writer.println("@attribute myclass {1,2,3,4,5}");
		arff_writer.println();
		arff_writer.println("@data");

		for (Map.Entry<String, List<String>> entry : mapLabels.entrySet())
		{
			List<String> classLabels = entry.getValue();
			Map<String, Long> map =  classLabels.stream()
					.collect(Collectors.groupingBy(w -> w, Collectors.counting()));

			List<Map.Entry<String, Long>> result = map.entrySet().stream()
					.sorted(Map.Entry.comparingByValue(Comparator.reverseOrder()))
					.limit(1)
					.collect(Collectors.toList());
			String[] groundTruth = result.get(0).toString().split("=");


			String sentenceKey = entry.getKey() + "_content";
			String sentenceData = mapSentences.get(sentenceKey);



			for(int j=0 ; j< uniqueWordList.size() ; j++)
			{
				if(sentenceData.contains(uniqueWordList.get(j)))
				{
					arff_writer.print("1,");
				}
				else
				{
					arff_writer.print("0,");
				}
			}
			int labelIndex = 0;
			for(int i=0;i<labels.size();i++)
				if(labels.get(i).equals(groundTruth[0])) labelIndex = i + 1;
			arff_writer.print(""+labelIndex);
			arff_writer.println();
		}

		arff_writer.close();
		arff_writer.flush();   

		return instancesData;
	}

	public static void main(String[] args) {

		KNNSentences tdta = new KNNSentences();
		try {

			String filePath = FileSystems.getDefault().getPath("src/sentenceData.arff").toAbsolutePath().toString();
			BufferedReader bReader = null;
			bReader = new BufferedReader(new FileReader(filePath));

			Instances train = new Instances (bReader);
			train.setClassIndex(train.numAttributes() - 1);

			bReader.close();

			IBk ibk = new IBk();
			ibk.buildClassifier(train);
			Evaluation eval = new Evaluation(train);
			eval.crossValidateModel(ibk, train, 10, new Random(1));
			System.out.println(eval.toSummaryString("\nResults\n=====\n", true));
			System.out.println("FMeasure is " + String.format("%.2f", eval.fMeasure(1) * 100) + "%");
			System.out.println("Precision is " + String.format("%.2f", eval.precision(1) * 100) + "%");
			System.out.println("Recall is " + String.format("%.2f", eval.recall(1) * 100) + "%");
			System.out.println("Confusion Matrix is " + eval.toMatrixString());
		} catch (Exception e) {
			System.err.println(e.getMessage());
			e.printStackTrace();
		}
	}
}