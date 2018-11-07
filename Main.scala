import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.clustering.{KMeans,KMeansModel}


object Main {
  def main(args : Array[String]): Unit ={
    Logger.getLogger("org").setLevel(Level.WARN)
    val appName = "NEO-K-Means"
    val master = "local[4]"
    val conf = new SparkConf().setAppName(appName).setMaster(master)
    val sc = new SparkContext(conf)
    val path = "C:\\Users\\Heedo\\Desktop\\data\\"

//    필요한 파라미터 입력
    val split_s = " "
    val k = 8

    val data = sc.textFile(path + "data.txt").map(line => line.split(split_s)).
      map(_.map(_.toDouble)).collect

    val N: Int = data.length



//    val U = sc.textFile(path + "U.txt").map(line => line.split(split_s)).map(_.map(_.toInt)).collect()

    val Init_data = sc.textFile(path + "data.txt").map{
       line => Vectors.dense(line.split(split_s).map(_.toDouble))
       }

    val U = Array.ofDim[Int](N, k)

    val test = KMeans.train(Init_data, k, 100)

    val clusterIndex = test.predict(Init_data).collect()
    for(i <- 0 until N){
      U(i)(clusterIndex(i)) = 1
    }


    val alpha = 0.1
    val beta = 0.005

    Test.test(sc, k, alpha, beta, data,U)
  }
}
