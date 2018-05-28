import java.io.BufferedReader;
import java.io.FileReader;
import java.nio.file.FileSystems;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.ClassificationViaRegression;
import weka.core.Instances;

public class EmailRochio {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		
		String filePath = FileSystems.getDefault().getPath("src/dbworld_bodies.arff").toAbsolutePath().toString();
		BufferedReader bReader = null;
		bReader = new BufferedReader(new FileReader(filePath));
		
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
		System.out.println("FMeasure is " + String.format("%.2f", evalRochio.fMeasure(1) * 100) + "%");
		System.out.println("Precision is " + String.format("%.2f", evalRochio.precision(1) * 100) + "%");
		System.out.println("Recall is " + String.format("%.2f", evalRochio.recall(1) * 100) + "%");
		System.out.println("Confusion Matrix is " + evalRochio.toMatrixString());
	}

}
