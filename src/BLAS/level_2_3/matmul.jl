import Base: *, A_mul_B!, At_mul_B!, Ac_mul_B!, Ac_mul_B, At_mul_B
import Base: \, A_ldiv_B!, At_ldiv_B!, Ac_ldiv_B!, Ac_ldiv_B, At_ldiv_B

const SparseMatrices{T} = Union{SparseMatrixCSC{T,BlasInt},
                                Symmetric{T,SparseMatrixCSC{T,BlasInt}},
                                LowerTriangular{T, SparseMatrixCSC{T,BlasInt}},
                                UnitLowerTriangular{T, SparseMatrixCSC{T,BlasInt}},
                                UpperTriangular{T, SparseMatrixCSC{T,BlasInt}},
                                UnitUpperTriangular{T, SparseMatrixCSC{T,BlasInt}}}

for T in [Complex{Float32}, Complex{Float64}, Float32, Float64]
for mat in (:StridedVector, :StridedMatrix)
for (trans, F!, F) in (('N', :A_mul_B! , :*),
                       ('C', :Ac_mul_B!, :Ac_mul_B),
                       ('T', :At_mul_B!, :At_mul_B))
    @eval begin
        function $F!(α::$T, A::SparseMatrixCSC{$T, BlasInt},
                       B::$mat{$T}, β::$T, C::$mat{$T})
            isa(B,AbstractVector) ?
                cscmv!($trans, α, matdescra(A), A, B, β, C) :
                cscmm!($trans, α, matdescra(A), A, B, β, C)
        end

        function $F!(C::$mat{$T}, A::SparseMatrices{$T},
                      B::$mat{$T})
            $F!(one($T), A, B, zero($T), C)
        end

        function $F(A::SparseMatrices{$T}, B::$mat{$T})
            isa(B,AbstractVector) ?
                $F!(zeros($T, size(A,1)),            A, B) :
                $F!(zeros($T, size(A,1), size(B,2)), A, B)
        end
    end

    for w in (:Symmetric, :LowerTriangular, :UnitLowerTriangular, :UpperTriangular, :UnitUpperTriangular)
        @eval begin
            function $F!(α::$T, A::$w{$T, SparseMatrixCSC{$T, BlasInt}},
                         B::$mat{$T}, β::$T, C::$mat{$T})
                isa(B,AbstractVector) ?
                    cscmv!($trans, α, matdescra(A), A.data, B, β, C) :
                    cscmm!($trans, α, matdescra(A), A.data, B, β, C)
            end
        end
    end
end

for (trans, F!, F) in (('N', :A_ldiv_B! , :\),
                       ('C', :Ac_ldiv_B!, :Ac_ldiv_B),
                       ('T', :At_ldiv_B!, :At_ldiv_B))
    for w in (:LowerTriangular, :UnitLowerTriangular, :UpperTriangular, :UnitUpperTriangular)
        @eval begin
            function $F!(α::$T, A::$w{$T, SparseMatrixCSC{$T, BlasInt}},
                           B::$mat{$T}, C::$mat{$T})
                isa(B,AbstractVector) ?
                    cscsv!($trans, α, matdescra(A), A.data, B, C) :
                    cscsm!($trans, α, matdescra(A), A.data, B, C)
            end

            function $F!(C::$mat{$T}, A::$w{$T, SparseMatrixCSC{$T, BlasInt}},
                         B::$mat{$T})
                $F!(one($T), A, B, C)
            end

            function $F(A::$w{$T, SparseMatrixCSC{$T, BlasInt}}, B::$mat{$T})
                isa(B,AbstractVector) ?
                    $F!(zeros($T, size(A,1)),            A, B) :
                    $F!(zeros($T, size(A,1), size(B,2)), A, B)
            end
        end
    end
end
end # mat
end # T
