python -m torch.distributed.launch --nproc_per_node=1 \
	main_task_retrieval.py --do_train --num_thread_reader=4 \
	--epochs=5 --batch_size=128 --n_display=50 \
	--features_path TODO \
	--output_dir ckpts/matchcut_frame \
	--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
	--datatype msrvtt \
	--feature_framerate 1 --coef_lr 1e-3 \
	--freeze_layer_num 0  --slice_framepos 2 \
	--loose_type --linear_patch 2d --sim_header seqTransf
