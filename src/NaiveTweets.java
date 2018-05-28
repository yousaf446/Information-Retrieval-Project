import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.nio.file.FileSystems;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.ClassificationViaRegression;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.EM;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.tokenizers.Tokenizer;
import weka.core.tokenizers.WordTokenizer;

public class NaiveTweets {

    /**
     * @param args the command line arguments
     */
    public static int countLinesNew(String filename) throws IOException 
    {
        InputStream is = new BufferedInputStream(new FileInputStream(filename));
        try {
            byte[] c = new byte[1024];

            int readChars = is.read(c);
            if (readChars == -1) {
                // bail out if nothing to read
                return 0;
            }

            // make it easy for the optimizer to tune this loop
            int count = 0;
            while (readChars == 1024) {
                for (int i=0; i<1024;) {
                    if (c[i++] == '\n') {
                        ++count;
                    }
                }
                readChars = is.read(c);
            }

            // count remaining characters
            while (readChars != -1) {
                for (int i=0; i<readChars; ++i) {
                    if (c[i] == '\n') {
                        ++count;
                    }
                }
                readChars = is.read(c);
            }

            return count == 0 ? 1 : count;
        } finally {
            is.close();
        }
    }
    
    public static void main(String[] args) throws Exception  {
        File SW_File = new File(FileSystems.getDefault().getPath("src/word_lists/stopwords.txt").toAbsolutePath().toString());
        List<String> swordArrayList = new ArrayList<String>();
        List<List<String>> docList = new ArrayList<List<String>>();
        List<List<Integer>> docFreqList = new ArrayList<List<Integer>>();
        List<Integer> docClassList = new ArrayList<Integer>();
        List<List<String>> docTestList = new ArrayList<List<String>>();
        List<List<Integer>> docTestFreqList = new ArrayList<List<Integer>>();
        List<Integer> docTestClassList = new ArrayList<Integer>();
        FileReader SW_fr=null;        
        try 
        {
            SW_fr =  new FileReader(SW_File.toString());
        } catch (FileNotFoundException ex) {
            Logger.getLogger(NaiveTweets.class.getName()).log(Level.SEVERE, null, ex);
        }
        BufferedReader br2 = new BufferedReader(SW_fr);
        String Rline2 =null;
        while ((Rline2 = br2.readLine() ) != null) 
        {       
            swordArrayList.add(Rline2);            
        }
        SW_fr.close();
        
        FilenameFilter filter = new FilenameFilter() {
        public boolean accept(File dir, String name) {
            return name.endsWith(".txt");
            }
        };
        
        File folder_dest = new File(FileSystems.getDefault().getPath("src/Health-Tweets/pre1/").toAbsolutePath().toString());
        File folder = new File(FileSystems.getDefault().getPath("src/Health-Tweets").toAbsolutePath().toString());
        File[] listOfFiles = folder.listFiles(filter);
        List<String> wordArrayList = new ArrayList<String>();
        List<String> subDocList = new ArrayList<String>();
        List<Integer> subDocFreqList = new ArrayList<Integer>();
        List<String> subTestDocList = new ArrayList<String>();
        List<Integer> subTestDocFreqList = new ArrayList<Integer>();
        for (int i = 0; i < listOfFiles.length; i++) 
        {
            File file = listOfFiles[i];
            File file_dest = new File (folder_dest.toString()+"/"+file.getName());
            int numOfLines = countLinesNew(file.toString());
            int trainLines = (numOfLines*5)/100;
            PrintWriter writer = new PrintWriter(file_dest.toString(), "UTF-8");
            FileReader fr = null;
            try {
                fr =  new FileReader(file.toString());
            } catch (FileNotFoundException ex) {
                Logger.getLogger(NaiveTweets.class.getName()).log(Level.SEVERE, null, ex);
            }
            BufferedReader br = new BufferedReader(fr);
            String Rline = null;
            while ((Rline = br.readLine() ) != null) 
            {
                try 
                {
                    String line[] = Rline.replace("|", "=").split("=");
                    String line2[] = line[2].split("http");
                    WordTokenizer wordTokenizer = new WordTokenizer();
                    wordTokenizer.tokenize(line2[0].replaceAll("[^a-zA-Z\\s]", ""));
                    while(wordTokenizer.hasMoreElements()) 
                    {
                        String word = wordTokenizer.nextElement();
                        String wordCompare = word.toLowerCase();
                        if(!swordArrayList.contains(wordCompare) && !wordCompare.contains("@") && !wordCompare.contains("#") )
                        {
                            if(trainLines > 0)
                            {
                                if(!wordArrayList.contains(word.toLowerCase()))
                                {
                                    wordArrayList.add(word.toLowerCase());
                                }
                                if(!subDocList.contains(word.toLowerCase()))
                                {
                                    subDocList.add(word.toLowerCase());
                                    subDocFreqList.add(1);
                                }
                                else
                                {
                                    subDocFreqList.set(subDocList.indexOf(word.toLowerCase()),(subDocFreqList.get(subDocList.indexOf(word.toLowerCase()))+1));
                                }
                            }
                            else if(trainLines <= 0 && trainLines >= -5)
                            {
                                if(!subDocList.contains(word.toLowerCase()))
                                {
                                    subDocList.add(word.toLowerCase());
                                    subDocFreqList.add(1);
                                }
                                else
                                {
                                    subDocFreqList.set(subDocList.indexOf(word.toLowerCase()),(subDocFreqList.get(subDocList.indexOf(word.toLowerCase()))+1));
                                }
                            }
                            writer.println(word.toLowerCase());
                        }
                    }
                    docList.add(new ArrayList<String>(subDocList));
                    docFreqList.add(new ArrayList<Integer>(subDocFreqList));
                    docClassList.add(i);
                    subDocFreqList.clear();
                    subDocList.clear();
                    trainLines--;
                    if(trainLines <-5)
                    {
                        break;
                    }
                } catch ( Exception ex) 
                {
                    Logger.getLogger(NaiveTweets.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
            fr.close();
            writer.close();
            writer.flush();                
        }
        File arff_file = new File(FileSystems.getDefault().getPath("src/Health-Tweets/pre1/data.arff").toAbsolutePath().toString());
        PrintWriter arff_writer = new PrintWriter(arff_file.toString(), "UTF-8");
        arff_writer.println("@relation health_care");
        arff_writer.println();
        for(int i=0 ; i< wordArrayList.size() ; i++)
        {
            arff_writer.println("@attribute "+wordArrayList.get(i)+" numeric");
            //System.out.println(i+"Word: "+docList.get(0).get(i)+" Frequency: "+docFreqList.get(0).get(i));
        }
        arff_writer.println("@attribute myclass {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}");
        arff_writer.println();
        arff_writer.println("@data");
        for(int i=0 ; i< docList.size() ; i++)
        {
            for(int j=0 ; j< wordArrayList.size() ; j++)
            {
                if(docList.get(i).contains(wordArrayList.get(j)))
                {
                    arff_writer.print(docFreqList.get(i).get(docList.get(i).indexOf(wordArrayList.get(j)))+",");
                }
                else
                {
                    arff_writer.print("0,");
                }
            }
            
            arff_writer.print(""+docClassList.get(i));
            arff_writer.println();
        }
        arff_writer.close();
        arff_writer.flush(); 
        
        String filePath = FileSystems.getDefault().getPath("src/Health-Tweets/pre1/data.arff").toAbsolutePath().toString();
        
        // Put that file in weka software & apply NaiveBayesMultinomial
        
        BufferedReader bReader = null;
		bReader = new BufferedReader(new FileReader(filePath));
		
		/*Instances dataset = new Instances (bReader);
		
		int trainSetSize = Math.round((dataset.numInstances() * 70)/100);
		int testSetSize = dataset.numInstances() - trainSetSize;
		
		Instances train = new Instances(dataset, 0, trainSetSize);
		train.setClassIndex(train.numAttributes() - 1);
		Instances test = new Instances(dataset, trainSetSize, testSetSize);
		test.setClassIndex(test.numAttributes() - 1);*/
		
		Instances train = new Instances (bReader);
		train.setClassIndex(train.numAttributes() - 1);
		
		bReader.close();
		
		IBk ibk = new IBk();
		ibk.setKNN(3);
		ibk.buildClassifier(train);
		Evaluation evalIBK = new Evaluation(train);
		evalIBK.crossValidateModel(ibk, train, 10, new Random(1));
		System.out.println(evalIBK.toSummaryString("\nResults\n=====\n", true));
		System.out.println("KNN FMeasure is " + String.format("%.2f", evalIBK.fMeasure(1) * 100) + "%");
		System.out.println("KNN Precision is " + String.format("%.2f", evalIBK.precision(1) * 100) + "%");
		System.out.println("KNN Recall is " + String.format("%.2f", evalIBK.recall(1) * 100) + "%");
		System.out.println("KNN Confusion Matrix is " + evalIBK.toMatrixString());
		
		NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(train);
		Evaluation evalNaive = new Evaluation(train);
		evalNaive.evaluateModel(nb, train);
		System.out.println(evalNaive.toSummaryString("\nResults\n=====\n", true));
		System.out.println("Naive Bayes FMeasure is " + String.format("%.2f", evalNaive.fMeasure(1) * 100) + "%");
		System.out.println("Naive Bayes Precision is " + String.format("%.2f", evalNaive.precision(1) * 100) + "%");
		System.out.println("Naive Bayes Recall is " + String.format("%.2f", evalNaive.recall(1) * 100) + "%");
		System.out.println("Naive Bayes Confusion Matrix is " + evalNaive.toMatrixString());
		
		ClassificationViaRegression rochio = new ClassificationViaRegression();
		rochio.buildClassifier(train);
		Evaluation evalRochio = new Evaluation(train);
		evalRochio.crossValidateModel(rochio, train, 10, new Random(1));
		System.out.println(evalRochio.toSummaryString("\nResults\n=====\n", true));
		System.out.println("FMeasure is " + String.format("%.2f", (evalRochio.fMeasure(1) * 100)) + "%");
		System.out.println("Precision is " + String.format("%.2f", (evalRochio.precision(1) * 100)) + "%");
		System.out.println("Recall is " + String.format("%.2f", (evalRochio.recall(1) * 100)) + "%");
		System.out.println("Confusion Matrix is " + evalRochio.toMatrixString());
		
		
		
		/*NaiveBayesMultinomial nb = new NaiveBayesMultinomial();
		nb.buildClassifier(train);
		Evaluation eval = new Evaluation(test);
		eval.evaluateModel(nb, test);
		System.out.println(eval.toSummaryString("\nResults\n=====\n", true));
		System.out.println("FMeasure is " + String.format("%.2f", eval.fMeasure(1) * 100) + "%");
		System.out.println("Precision is " + String.format("%.2f", eval.precision(1) * 100) + "%");
		System.out.println("Recall is " + String.format("%.2f", eval.recall(1) * 100) + "%");
		System.out.println("Confusion Matrix is " + eval.toMatrixString());*/
		
    }    
}
