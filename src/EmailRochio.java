import java.io.BufferedReader;
import java.io.FileReader;
import java.nio.file.FileSystems;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
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
		
		ClassificationViaRegression rochio = new ClassificationViaRegression();
		rochio.buildClassifier(train);
		Evaluation eval = new Evaluation(train);
		eval.crossValidateModel(rochio, train, 10, new Random(1));
		System.out.println(eval.toSummaryString("\nResults\n=====\n", true));
		System.out.println("FMeasure is " + String.format("%.2f", eval.fMeasure(1) * 100) + "%");
		System.out.println("Precision is " + String.format("%.2f", eval.precision(1) * 100) + "%");
		System.out.println("Recall is " + String.format("%.2f", eval.recall(1) * 100) + "%");
		System.out.println("Confusion Matrix is " + eval.toMatrixString());
	}

}
