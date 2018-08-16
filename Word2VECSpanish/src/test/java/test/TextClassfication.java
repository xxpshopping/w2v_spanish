package test;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.SolverType;

public class TextClassfication {
	Feature[][] featureMatrix;
	Feature[][] testfeatureMatrix;
	double[] trainTarget;
	double[] testTarget;
	ArrayList<Feature[]> trainFeatureList = new ArrayList<Feature[]>();
	ArrayList<Feature[]> testFeatureList = new ArrayList<Feature[]>();
	ArrayList<Double> trainTargetList = new ArrayList<Double>();
	ArrayList<Double> testTargetList = new ArrayList<Double>();
	int trainNumber;
	int testNumber;
	//���Լ���û�������
	double[] typeSize;
	//Ԥ��õ���ÿ������
	double[] preSize;
	File[] categories;
	String[] classes = {"P", "N"};
	
	int dim = 100;
	Map<String, Integer> words = new HashMap<>();
	ArrayList<ArrayList<Double>> C = new ArrayList<ArrayList<Double>>();
	String trainfile;
	boolean ismulu;
	
	public TextClassfication(String trainfile, boolean ismulu) {
		this.trainfile = trainfile;
		this.ismulu = ismulu;
	}

	public void getDocVec(String textPath, String encode) {
		try {
			if (ismulu) {
				File dir = new File(textPath);
				categories = dir.listFiles();
				Arrays.sort(categories);
				int countOfCategory = categories.length;
				preSize = new double[countOfCategory];
				typeSize = new double[countOfCategory];
				String[] tokens;
				
				for (int k = 0; k < countOfCategory; k++) {
					File[] files = categories[k].listFiles();
					Arrays.sort(files);
					int countOfTestType = 0;
					for (int f = 0; f < files.length; f++) {
						File[] contents = files[f].listFiles();
						Arrays.sort(contents);
						int countOfFile = contents.length;

						for (int i = 0; i < countOfFile; i++) {
							// get the text to the String
							InputStreamReader input = new InputStreamReader(new FileInputStream(contents[i]), encode);
							BufferedReader read = new BufferedReader(input);
							String line;
							String text = "";
							int linecount = 0;
							while ((line = read.readLine()) != null) {
								// text = text + line;
								// }
								tokens = line.split(" ");
								linecount++;

								// get the average Vector as document vector
								int countOfWords = 0;
								double[] docVec = new double[dim];
								int j, k1;
								for (int wordcount = 0; wordcount < tokens.length; wordcount++) {

									for (j = 0; j < tokens[wordcount].length();) {
										if (tokens[wordcount].substring(j, j + 1)
												.matches("[\\pP+~$`^=|<>～｀＄＾＋＝｜＜＞￥×]")) {
											j++;
										} else {
											break;
										}
									}
									if (j < tokens[wordcount].length()) {
										for (k1 = tokens[wordcount].length() - 1; k1 >= 0;) {
											if (tokens[wordcount].substring(k1, k1 + 1)
													.matches("[\\pP+~$`^=|<>～｀＄＾＋＝｜＜＞￥×]")) {
												k1--;
											} else {
												break;
											}
										}
										tokens[wordcount] = tokens[wordcount].substring(j, k1 + 1);
									}
									if (tokens[wordcount].matches("[a-zA-Z'-]+")) {
										Integer index = words.get(tokens[wordcount]);
										if (index != null) {
											countOfWords++;
											for (int d = 0; d < dim; d++) {
												docVec[d] += C.get(index).get(d);
											}
										}
									}

								}

								if (countOfWords != 0) {
									Feature[] feature = new Feature[dim];
									for (j = 0; j < dim; j++) {
										docVec[j] /= countOfWords;
										feature[j] = new FeatureNode(j + 1, docVec[j]);
									}
									if (linecount < 5331 * 0.8) {
										trainFeatureList.add(feature);
										trainTargetList.add((double) k);
									} else {
										countOfTestType++;
										testFeatureList.add(feature);
										testTargetList.add((double) k);
									}
								}
							}
						}
					}
					typeSize[k] = countOfTestType;
				}
			} else {
				InputStreamReader testinput = new InputStreamReader(new FileInputStream(textPath + "/test"), encode);
				BufferedReader testread = new BufferedReader(testinput);
				InputStreamReader traininput = new InputStreamReader(new FileInputStream(textPath + "/train"), encode);
				BufferedReader trainread = new BufferedReader(traininput);
				
				BufferedReader[] readers = {trainread, testread};
				
				
				Map<String, Integer> classesSize = new HashMap<>();
				classesSize.put("P", 0);
				classesSize.put("N", 0);
				//classesSize.put("NONE", 0);
				//classesSize.put("NEU", 0);
				Map<String, Integer> classesIndex = new HashMap<>();
				classesIndex.put("P", 0);
				classesIndex.put("N", 1);
				//classesIndex.put("NONE", 2);
				//classesIndex.put("NEU", 3);
				
				String line;
				typeSize = new double[2];
				preSize = new double[2];
				String label;
				String[] tokens;
				for (int r = 0; r < readers.length; r++) {
					while ((line = readers[r].readLine()) != null) {
						label = line.split(" ")[0];
						if(label.equals("NONE") || label.equals("NEU")) {
							continue;
						}
						tokens = line.split(" ");

						// get the average Vector as document vector
						int countOfWords = 0;
						double[] docVec = new double[dim];
						int j, k1;
						for (int wordcount = 1; wordcount < tokens.length; wordcount++) {

							for (j = 0; j < tokens[wordcount].length();) {
								if (tokens[wordcount].substring(j, j + 1).matches("[\\pP+~$`^=|<>～｀＄＾＋＝｜＜＞￥×]")) {
									j++;
								} else {
									break;
								}
							}
							if (j < tokens[wordcount].length()) {
								for (k1 = tokens[wordcount].length() - 1; k1 >= 0;) {
									if (tokens[wordcount].substring(k1, k1 + 1).matches("[\\pP+~$`^=|<>～｀＄＾＋＝｜＜＞￥×]")) {
										k1--;
									} else {
										break;
									}
								}
								tokens[wordcount] = tokens[wordcount].substring(j, k1 + 1);
							}
							if (!tokens[wordcount].equals(" ") && !tokens[wordcount].equals("\t") && !tokens[wordcount].equals("")) {
								
								Integer index = words.get(tokens[wordcount]);
								if (index != null) {
									countOfWords++;
									for (int d = 0; d < dim; d++) {
										docVec[d] += C.get(index).get(d);
									}
								}
							}
						}

						if (countOfWords != 0) {
							
							Feature[] feature = new Feature[dim];
							for (j = 0; j < dim; j++) {
								docVec[j] /= countOfWords;
								feature[j] = new FeatureNode(j + 1, docVec[j]);
							}
							if(r == 0) {
								trainFeatureList.add(feature);
								trainTargetList.add((double) classesIndex.get(label));
							} else {
								classesSize.put(label, classesSize.get(label) + 1);
								testFeatureList.add(feature);
								testTargetList.add((double) classesIndex.get(label));
							}
						}
					}
				}
				for(int i = 0; i < classes.length; i++) {
					typeSize[classesIndex.get(classes[i])] = classesSize.get(classes[i]);
				}
				for(int i = 0; i < typeSize.length; i++) {
					System.out.println(typeSize[i]);
				}
			}

			trainNumber = trainFeatureList.size();
			testNumber = testFeatureList.size();
			System.out.println(trainNumber + " " + testNumber);
			featureMatrix = new Feature[trainNumber][];
			testfeatureMatrix = new Feature[testNumber][];
			trainTarget = new double[trainNumber];
			testTarget = new double[testNumber];

			for (int i1 = 0; i1 < trainNumber; i1++) {
				featureMatrix[i1] = trainFeatureList.get(i1);
				trainTarget[i1] = trainTargetList.get(i1);
			}

			for (int i1 = 0; i1 < testNumber; i1++) {
				testfeatureMatrix[i1] = testFeatureList.get(i1);
				testTarget[i1] = testTargetList.get(i1);
			}
			testFeatureList.clear();
			testTargetList.clear();
			trainFeatureList.clear();
			trainTargetList.clear();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public void readVec() {
		try {
			InputStreamReader input = new InputStreamReader(new FileInputStream(trainfile), "utf-8");
			BufferedReader read = new BufferedReader(input);
			String line;
			String[] factors;
			int num = 0;
			while((line = read.readLine()) != null) {
				if(line.equals("the word vector is ")) {
					continue;
				}
				if(line.equals("the pinyin vector is ") || line.equals("the Character vector is ")) {
					break;
				}
				factors = line.split(" ");
				words.put(factors[0], num);
				ArrayList<Double> vec = new ArrayList<Double>();
				for(int i = 1; i <= dim; i++) {
					vec.add(Double.valueOf(factors[i]));
				}
				C.add(vec);
				num++;
			}
			System.out.println("C size " + C.size() + " " + num);
		} catch (UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public double calculateAcc(String textPath, String encode, BufferedWriter write) throws IOException {
		// loading train data
		readVec();
		getDocVec(textPath, encode);

		Problem problem = new Problem();
		problem.l = trainNumber; // number of training examples��ѵ��������
		problem.n = dim; // number of features������ά��
		problem.x = featureMatrix; // feature nodes����������
		problem.y = trainTarget; // target values�����

		SolverType solver = SolverType.L2R_L2LOSS_SVC_DUAL; // -s 1
		double C = 1.0; // cost of constraints violation
		double eps = 0.01; // stopping criteria

		Parameter parameter = new Parameter(solver, C, eps);
		Model model = Linear.train(problem, parameter);
		File modelFile = new File("model");
		int i = 0;
		int j = 0;
		//ÿ����������ȷ������
		double[] accurateType = new double[typeSize.length];
		try {
			model.save(modelFile);
			// load model or use it directly
			model = Model.load(modelFile);

			for (j = 0; j < testfeatureMatrix.length; j++) {
				double prediction = Linear.predict(model, testfeatureMatrix[j]);
				preSize[(int) prediction]++;
				if(prediction == testTarget[j]) {
					accurateType[(int) prediction]++;
					i++;
				}
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		double[] acc = new double[typeSize.length];
		double[] recall = new double[typeSize.length];
		double[] f = new double[typeSize.length];
		for (int typeIndex = 0; typeIndex < typeSize.length; typeIndex++) {
			acc[typeIndex] = accurateType[typeIndex] / preSize[typeIndex] * 100;
			recall[typeIndex] = accurateType[typeIndex] / typeSize[typeIndex] * 100;
			f[typeIndex] = 2 * acc[typeIndex] * recall[typeIndex] / (acc[typeIndex] + recall[typeIndex]);
			if(ismulu) {
				write.write(categories[typeIndex].getName() + " " + String.format("%.4f", acc[typeIndex]) + " " + String.format("%.4f", recall[typeIndex]) + " " + String.format("%.4f", f[typeIndex]) + " ");
				write.flush();
			} else {
				write.write(classes[typeIndex] + " " + String.format("%.4f", acc[typeIndex]) + " " + String.format("%.4f", recall[typeIndex]) + " " + String.format("%.4f", f[typeIndex]) + " ");
				write.flush();
			}
		}
		return (double) i / (double) testNumber * 100;
	}
}
