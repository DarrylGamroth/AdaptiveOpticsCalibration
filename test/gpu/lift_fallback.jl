function test_accelerator_lift_fallback(backend, to_device, device_array_type)
    @testset "LiFT rank-deficient device solve fallback" begin
        T = Float32
        jacobian = T[1 0; 2 0; 3 0]
        residual = T[2, 4, 6]
        normal = transpose(jacobian) * jacobian
        rhs = transpose(jacobian) * residual
        model = AcceleratorLinearLiFTModel(
            to_device(T[10, 12, 8]), to_device(jacobian))
        method = PR.LiFT(iterations=1,
            jacobian_method=PR.LiFTAnalyticJacobian(),
            solve_mode=PR.LiFTSolveNormalEquations(),
            model_scaling=PR.LiFTPhysicalRatePreservation(),
            check_convergence=false)
        plan = prepare(method, KernelExecution(
            PR.LiFTSpecification(model, PR.LiFTPhotonRate()), backend))
        workspace = allocate_workspace(plan)
        inverse = workspace.inverse
        # The advanced device-solve seam accepts explicit loading. Zero loading
        # leaves this normal matrix singular through all Cholesky attempts and
        # exercises SVD, including its zero-singular-value denominator branch.
        # Positive loading instead takes the regularized Cholesky retry.
        for loading in (zero(T), T(0.125))
            copyto!(inverse.jacobian, jacobian)
            copyto!(inverse.residual, residual)
            copyto!(inverse.normal, normal)
            copyto!(inverse.rhs, rhs)
            regularization, used_fallback = PR.solve_lift_normal_device!(
                workspace, plan, PR.LiFTDampingNone(), zero(T), loading)
            expected = iszero(loading) ?
                pinv(Float64.(jacobian)) * Float64.(residual) :
                (Float64.(normal) + Float64(loading) * I) \ Float64.(rhs)
            @test inverse.rhs isa device_array_type
            @test used_fallback == iszero(loading)
            @test regularization == loading
            @test Array(inverse.rhs) ≈ expected rtol=2f-5 atol=2f-6
            @test all(isfinite, Array(inverse.rhs))
            @test Array(inverse.jacobian) == jacobian
            @test Array(inverse.normal) == normal
            @test Array(inverse.residual) == residual
        end
    end
end
