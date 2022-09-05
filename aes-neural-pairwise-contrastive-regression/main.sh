# @Author: kaiwei cai

checkdir=./checkpoint

if [ "$#" -ne 4 ]; then
  echo "Usage: sh $0 embed_dim prompt_id fold_id embed_type"
  exit 1
fi


embed_dim=$1
prompt_id=$2
fold_id=$3
embed_type=$4


if [ "$#" -ne 4 ]; then
  echo "Usage: sh $0 embed_dim prompt_id fold_id embed_type"
  exit 1
fi


embed_dim=$1
prompt_id=$2
fold_id=$3
embed_type=$4

echo $embed_type

datadir=./../data-set/asap/fold_${fold_id}
trainfile=$datadir/train.tsv
devfile=$datadir/dev.tsv
testfile=$datadir/test.tsv

if [ ! -d $checkdir/preds ]; then
	mkdir -p $checkdir/preds
fi

if [ $embed_type = "word2vec" ]
then
	embed_dir=../data/embedding
	embeddingfile=$embed_dir/data_dim${embed_dim}.vec
elif [ $embed_type = "glove" ]
then
	embed_dir=embedding/glove.6B
	embeddingfile=$embed_dir/glove.6B.${embed_dim}d.txt.gz
fi

nb_epochs=80
# echo $embed_dim

#echo "Using embedding ${embeddingfile}"

 # THEANO_FLAGS='floatX=float32,device=cpu'
 export CUDA_VISIBLE_DEVICES=2
 python main.py --fine_tune --num_epochs $nb_epochs --batch_size 6 \
 --nbfilters 100 --filter1_len 5 --filter2_len 3 --rnn_type LSTM --lstm_units 100\
 	--optimizer rmsprop --learning_rate 0.00001 --dropout 0.5  \
	--oov embedding  --checkpoint_path $checkdir \
	--train $trainfile --dev $devfile --test $testfile --prompt_id $prompt_id \
	--train_flag --mode att --example_size 50 #--init_bias #--l2_value 0.001
