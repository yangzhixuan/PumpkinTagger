require("words_stream.jl")

type Tagger
    feature_id :: Dict{ String, Int }
    label_id :: Dict{ String, Int }
    label_name :: Dict{ Int, String }
    weights :: Array{ Float64, 2 }

    feature_count :: Int
    label_count :: Int
end

function Tagger()
    Tagger( Dict(), Dict(), Dict(), zeros(0,0), 0, 0 )
end

function init_tagger(tagger :: Tagger, file :: String)
    @printf "reading the training set...\n"
    feature_count = 0
    label_count = 0
    cached = collect(WordsStream(file))
    cached :: Array{WordInstance}
    for w in cached
        if !haskey(tagger.label_id, w.label)
            label_count += 1
            tagger.label_id[w.label] = label_count
            tagger.label_name[label_count] = w.label
        end
        for (f, s) in w.features
            if !haskey(tagger.feature_id, f)
                feature_count += 1
                tagger.feature_id[f] = feature_count
            end
        end
    end
    @printf "%d features and %d labels in the training set\n" feature_count label_count
    tagger.feature_count = feature_count

    for (l1,id) in tagger.label_id
        for (l2, id) in tagger.label_id
            feature_count += 1
            tagger.feature_id[l1 * l2] = feature_count
        end
    end

    tagger.label_count = label_count
    tagger.weights = zeros(label_count, feature_count)
    cached
end

function train_tagger(tagger :: Tagger, file :: String; max_iter = 10)
    # we store the training set in the memory for efficiency
    cached = init_tagger(tagger, file)
    cached :: Array{ WordInstance }

    scores = zeros(tagger.label_count)
    last_failed = typemax(Int)

    weights = zeros(size(tagger.weights))
    for t in 1:max_iter
        @printf "start iteration %d ... \n" t
        failed = 0
        success = 0
        last_tag = [ "__START__", "__START__" ]
        tic()
        for i in 1:length(cached)
            w = cached[i]
            fill!(scores, 0.0)
            if w.sentence_start != true && (i > 1 && cached[i-1].sentence_start != true)
                # first order linear chain structure
                push!(w.features, (last_tag[2 - ((i - 2) & 1)] * last_tag[2 - ((i-1) & 1)], 1.0))
            end

            for (f, s) in w.features
                fid = tagger.feature_id[f]
                for l in 1:tagger.label_count
                    scores[l] += s * tagger.weights[l, fid]
                end
            end

            last_tag[2 - (i & 1)] = w.label
            (max_score, z) = findmax(scores)
            y = tagger.label_id[w.label]

            if y != z
                # update the weights if the predication is not correct
                failed += 1
                for (f,s) in w.features
                    fid = tagger.feature_id[f]
                    tagger.weights[y, fid] += s
                    tagger.weights[z, fid] -= s
                end
            else
                success += 1
            end
        end
        running_time = toq()
        @printf "iteration %d: %d successfully predicated, %d misprediction (accuracy: %f, %f seconds elapsed)\n" t success failed (success / (success + failed)) running_time
        weights += tagger.weights
        if failed == 0 || (failed > last_failed)
            break
        end
        last_failed = failed
    end
    tagger.weights = weights
    tagger
end

abstract LabelSequence
type Null <: LabelSequence
end
nullnode = Null()
type Node <: LabelSequence
    label :: String
    next :: LabelSequence
end

function collect_labels( l :: LabelSequence )
    a = String[]
    while l != nullnode
        push!(a, l.label)
        l = l.next
    end
    reverse!(a)
    a
end

function decode(tagger :: Tagger, file :: String; report_accuracy = false)
    @printf "reading %s ...\n" file
    cached = collect(WordsStream(file))
    cached :: Array{ WordInstance }

    f = zeros(tagger.label_count, tagger.label_count, 2)
    g = Array(LabelSequence, tagger.label_count, tagger.label_count, 2)
    fill!(g, nullnode)

    label_fid = Array(Int, tagger.label_count, tagger.label_count)
    for (l1, id1) in tagger.label_id
        for (l2, id2) in tagger.label_id
            label_fid[id1, id2] = tagger.feature_id[l1 * l2]
        end
    end

    @printf "start decoding...\n"
    tic()
    for i in 1:length(cached)
        if i % 1000 == 0
            @printf "%d words decoded, progress %f\n" i (i / length(cached))
        end

        inx = 2 - i & 1
        pinx = 2 - (i - 1) & 1

        for l in 1:tagger.label_count
            for l2 in 1:tagger.label_count
                f[l, l2, inx] = 0
                g[l, l2, inx] = Node(tagger.label_name[l], nullnode)
            end
        end

        # scores for features in i-th position
        for (fea, scale) in cached[i].features
            if haskey(tagger.feature_id, fea)
                fid = tagger.feature_id[fea]
                for l in 1:tagger.label_count
                    for l2 in 1:tagger.label_count
                        f[l, l2, inx] += tagger.weights[l, fid] * scale
                    end
                end
            end
        end

        # enumerate every possible previous labels ( Viterbi algorihtm )
        for l in 1:tagger.label_count
            for l2 in 1:tagger.label_count
                max_score = -Inf
                max_pl = 0
                for pl in 1:tagger.label_count
                    s = f[l2, pl, pinx]
                    if !cached[i].sentence_start && (i > 1 && !cached[i-1].sentence_start)
                        s += tagger.weights[ l, label_fid[pl, l2] ]
                    end
                    if s > max_score
                        max_pl = pl
                        max_score = s
                    end
                end
                f[l, l2, inx] += max_score
                g[l, l2, inx].next = g[l2, max_pl, pinx]
            end
        end
    end
    toc()
    (max_score, l) = findmax(f[:, :, 2 - length(cached) & 1])
    result = collect_labels(g[:, :, 2 - length(cached) & 1][l])

    if report_accuracy
        succ = 0
        for i in 1:length(result)
            if cached[i].label == result[i]
                succ += 1
            end
        end
        @printf "accuracy on %s: %f\n" file (succ / length(result))
    end
    return result
end
