import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.linalg.{Matrices, Matrix}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, RowMatrix}

class Test private(
                    private var sc : SparkContext,
                    private val k : Int,
                    private var alpha : Double,
                    private var beta : Double,
                    private var max_iter : Int = 100){

  def run_algorithm(
                     X : Array[Array[Double]],
                     initU : Array[Array[Int]]): Unit ={

    val task_num = 8

    //변수 초기화
    val N = X.length
    val dim = X(0).length
    var t =0
    val t_max= max_iter
    val alphaN :Int= (alpha * N).toInt
    val betaN : Int= (beta * N).toInt

    var J : Double= Double.MaxValue
    var old_J : Double =0.0
    val epsilon = 0.0


    //RDD[(Int, Array[Double])] : 데이터 번호와 데이터 좌표
    val X_rdd = sc.parallelize(X.zipWithIndex.map{case(x,y) => (y.toLong,x)}, task_num).persist
    //데이터에 대한 행렬 [N x dim]
    val X_mat = Matrices.dense(N, dim, X.transpose.flatten)
    //assignment 행렬 [k x N]
    var U = initU.transpose
    //클러스터의 중심과 데이터 사이의 거리를 저장하는 행렬 [N x k]
    // val D = Array.ofDim[Double](N,k)


    val iterationStartTime = System.nanoTime()

    //계산 시작
    while ((old_J - J).abs > epsilon && t <= t_max){
      old_J = J
      J = 0



      //====================각 클러스터의 중심 구하기==================================================================

      //RDD[(index, Vector[Double]) : 데이터의 assignment 정보를 cluster 번호에 따라 나열한 것.
      val U_rdd = sc.parallelize(U.zipWithIndex.map{ case (x,y) => (y, x)}, task_num)
      val U_irdd = U_rdd.map{ case(x, y) => IndexedRow(x.toLong, Vectors.dense(y.map(_.toDouble))) }
      val U_mat = new IndexedRowMatrix(U_irdd)


      //각 클러스터에 할당된 데이터의 개수
      val count_rdd = U_irdd.map(x => {
        var count = 0.0
        x.vector.foreachActive( (i,v) => count += v)
        (x.index, count)
      })

      val M_mat = U_mat.multiply(X_mat)

      //RDD(index, Vector[Double]) : 클러스터 번호와 클러스터의 중심들을 저장.
      val M_rdd = M_mat.rows.cartesian(count_rdd).filter(x => x._1.index == x._2._1).
        map{ case(x,y) => (x.index, Vectors.dense(x.vector.toArray.map(x => x/y._2))) }


      //====================각 클러스터의 중심 과 데이터들 사이의 거리 구하기==========================================

      val D_rdd = X_rdd.cartesian(M_rdd).map{
        case (x, m) => {
          var dif = 0.0
          var d = 0.0

          val ix = x._2.iterator
          val im = m._2.toArray.iterator
          while(ix.hasNext) {
            dif = im.next - ix.next
            d += dif * dif
          }
          (d, x._1, m._1)
        }
      }.persist

      val sorted_dnk_rdd = D_rdd.map(x => (x._2 , (x._3, x._1))).reduceByKey((V1, V2) => {
        if (V1._2 < V2._2)
          V1
        else
          V2
      }).map{ case(x,y) => (y._2, x, y._1)}.sortBy(_._1)

      //====================(N - betaN)개의 데이터를 클러스터에 할당하기===============================================


      var numAssign: Int = N - betaN

      var sorted_dnk = sorted_dnk_rdd.take(numAssign)
      U = Array.ofDim[Int](k, N)
      val D = Array.ofDim[(Double, Long, Long)](numAssign)

      for(i <- 0 until numAssign){
        J += sorted_dnk(i)._1
        U(sorted_dnk(i)._3.toInt)(sorted_dnk(i)._2.toInt) = 1
        D(i) = sorted_dnk(i)
      }

      val sub_D_rdd = sc.parallelize(D)
      val D2_rdd = D_rdd.subtract(sub_D_rdd).sortBy(_._1)

      //====================(alphaN + betaN)개의 데이터를 클러스터에 할당하기===========================================

      numAssign = alphaN + betaN

      sorted_dnk = D2_rdd.take(numAssign)

      for(i <- 0 until numAssign){
        J += sorted_dnk(i)._1
        U(sorted_dnk(i)._3.toInt)(sorted_dnk(i)._2.toInt) = 1
      }
      t += 1
      println(s"No. of iterations done: ${t}\n")


    }
    val iterationTime = (System.nanoTime()  - iterationStartTime) / 1e9


    //====================결과 비교==================================================================================

    if(k == 2){
      val ground = sc.textFile("C:\\Users\\Heedo\\Desktop\\data\\ground.txt").map(line => line.split("\t")).
        map(_.map(_.toInt)).collect

      var countA = 0
      var countB = 0
      U = U.transpose
      for(i <- 0 until N){
        if( (U(i)(0) == ground(i)(0))  && (U(i)(1) == ground(i)(1))) countA += 1
        if( (U(i)(1) == ground(i)(0))  && (U(i)(0) == ground(i)(1))) countB += 1
      }
      println(s"count A : ${countA}\ncount B : ${countB}\n")
    }


    println(s"No. of total iterations done: ${t}\n")
    println(s"Total objective : ${J}\n")
    println(s"Total time of iterations done: ${iterationTime}\n")
  }


}


object Test {

  def test(
            sc : SparkContext,
            k : Int,
            alpha : Double,
            beta : Double,
            X : Array[Array[Double]],
            initU : Array[Array[Int]]) = {

    new Test(sc, k, alpha, beta).run_algorithm(X,initU)

  }
}
