#Adapted from DFTK.jl: https://github.com/JuliaMolSim/DFTK.jl/blob/master/src/external/spglib.jl
# Updated to use Spglib.jl v1.x Julia API (no more direct ccall to spglib_jll)

# Routines for interaction with spglib via Spglib.jl v1.x
# Note: Spglib.jl v1.x follows the spglib/python convention (same as DFTK).
#       Rotations returned by get_symmetry are already transposed to Julia column-major.
import Spglib

"""
Wrapper around Spglib.standardize_cell return value.
Provides error messages when accessing deprecated v0.6 field names.
"""
struct StandardizedCell{C}
    cell::C
end

function Base.getproperty(w::StandardizedCell, s::Symbol)
    if s === :types
        error("`types` field was removed in Spglib v1.x. Use `atoms` instead.")
    elseif s === :numbers
        error("`numbers` field was removed in Spglib v1.x. Use `atoms` instead.")
    end
    getproperty(getfield(w, :cell), s)
end
Base.propertynames(w::StandardizedCell) = propertynames(getfield(w, :cell))

"""
Convert the atom groups and positions datastructure into a tuple of datastructures for
use with spglib. Validity of the input data is assumed. The output `positions` contains
positions per atom, `numbers` contains the mapping atom to a unique number for each group
of indistinguishable atoms, `spins` contains the ``z``-component of the initial magnetic
moment on each atom, and `collinear` whether the atoms mark a case of collinear spin or not.
Notice that if `collinear` is false then `spins` is garbage.
"""
function spglib_atoms(atom_groups,
    positions::AbstractVector{<:AbstractVector{<:AbstractFloat}},
    magnetic_moments)
    n_attypes = length(positions)
    spg_numbers = zeros(Cint, n_attypes)
    spg_spins = zeros(Cdouble, n_attypes)
    spg_positions = [zeros(Float64, 3) for _ in 1:n_attypes]

    arbitrary_spin = false
    offset = 0
    for (igroup, indices) in enumerate(atom_groups)
        for iatom in indices
            offset += 1
            spg_numbers[offset] = igroup
            spg_positions[offset][1:length(positions[iatom])] .= positions[iatom]

            if !isempty(magnetic_moments)
                magmom = magnetic_moments[iatom]
                spg_spins[offset] = magmom[3]
                !iszero(magmom[1:2]) && (arbitrary_spin = true)
            end
        end
    end

    collinear = !arbitrary_spin && !all(iszero, spg_spins)
    (; positions=spg_positions, numbers=spg_numbers, spins=spg_spins, collinear)
end

function spglib_cell(lattice, atom_groups, positions, magnetic_moments)
    spg = spglib_atoms(atom_groups, positions, magnetic_moments)
    (; cell=Spglib.Cell(lattice, spg.positions, spg.numbers, spg.spins), spg.collinear)
end
function spglib_cell(model, magnetic_moments)
    spglib_cell(model.lattice, model.atom_groups, model.positions, magnetic_moments)
end


function spglib_get_symmetry(lattice::AbstractMatrix{<:AbstractFloat}, atom_groups,
    positions, magnetic_moments=[];
    tol_symmetry=SYMMETRY_TOLERANCE)
    lattice = Matrix{Float64}(lattice)  # spglib operates in double precision

    if isempty(atom_groups)
        # spglib doesn't like no atoms, so we default to
        # no symmetries (even though there are lots)
        return [Mat3{Int}(I)], [Vec3(zeros(3))]
    end

    # Build SpglibCell and call Spglib.jl v1.x API
    cell, collinear = spglib_cell(lattice, atom_groups, positions, magnetic_moments)

    if collinear
        rotations_raw, translations_raw, _ = Spglib.get_symmetry_with_collinear_spin(cell, tol_symmetry)
    else
        rotations_raw, translations_raw = Spglib.get_symmetry(cell, tol_symmetry)
    end

    # Convert SMatrix/SVector to Mat3/Vec3
    # Note: Spglib.jl v1.x already transposes rotations to Julia column-major convention
    Ws = [Mat3{Int}(W) for W in rotations_raw]
    ws = [Vec3{eltype(lattice)}(w) for w in translations_raw]

    # Check (W, w) maps atoms to equivalent atoms in the lattice
    for (W, w) in zip(Ws, ws)
        # Check (A W A^{-1}) is orthogonal
        Wcart = lattice * W / lattice
        if maximum(abs, Wcart'Wcart - I) > tol_symmetry
            error("spglib returned bad symmetries: Non-orthogonal rotation matrix.")
        end

        for group in atom_groups
            group_positions = positions[group]
            for coord in group_positions
                # If all elements of a difference in diffs is integer, then
                # W * coord + w and pos are equivalent lattice positions
                if !any(c -> is_approx_integer(W * coord + w - c; tol=tol_symmetry), group_positions)
                    error("spglib returned bad symmetries: Cannot map the atom at position " *
                          "$coord to another atom of the same element under the symmetry " *
                          "operation (W, w):\n($W, $w)")
                end
            end
        end
    end

    return Ws, ws
end

# The irreducible k-points are searched from unique k-point mesh grids from direct (real space) basis vectors
# and a set of rotation parts of symmetry operations in direct space with one or multiple stabilizers.
function spglib_get_stabilized_reciprocal_mesh(kgrid_size, rotations::Vector;
    is_shift=Vec3(0, 0, 0),
    is_time_reversal=false,
    qpoints=[Vec3(0.0, 0.0, 0.0)])

    # Convert rotations to SMatrix format expected by Spglib.jl v1.x
    spg_rotations = [SMatrix{3,3,Int32,9}(Cint.(S)) for S in rotations]

    result = Spglib.get_stabilized_reciprocal_mesh(
        spg_rotations, kgrid_size, qpoints;
        is_shift=is_shift, is_time_reversal=is_time_reversal
    )

    # Return in the same format as before: (n_kpts, mapping, grid_address)
    # Note: result.ir_mapping_table is 1-indexed in v1.x
    n_kpts = length(unique(result.ir_mapping_table))
    mapping = Int.(result.ir_mapping_table)
    grid = [Vec3{Int}(ga) for ga in result.grid_address]
    return n_kpts, mapping, grid
end

normalize_magnetic_moment(::Nothing)::Vec3{Float64} = (0, 0, 0)
normalize_magnetic_moment(mm::Number)::Vec3{Float64} = (0, 0, mm)
normalize_magnetic_moment(mm::AbstractVector)::Vec3{Float64} = mm

"""
Returns crystallographic conventional cell according to the International Table of
Crystallography Vol A (ITA) in case `primitive=false`. If `primitive=true`
the primitive lattice is returned in the convention of the reference work of
Cracknell, Davies, Miller, and Love (CDML). Of note this has minor differences to
the primitive setting choice made in the ITA.
"""
function spglib_standardize_cell(lattice::AbstractArray{T}, atom_groups, positions,
    magnetic_moments=[];
    correct_symmetry=true, primitive=false,
    tol_symmetry=SYMMETRY_TOLERANCE) where {T}
    # TODO For time-reversal symmetry see the discussion in PR 496.
    #      https://github.com/JuliaMolSim/DFTK.jl/pull/496/files#r725203554
    #      Essentially this does not influence the standardisation,
    #      but it only influences the kpath.
    cell, _ = spglib_cell(lattice, atom_groups, positions, magnetic_moments)
    std_cell = Spglib.standardize_cell(cell, tol_symmetry;
        to_primitive=primitive, no_idealize=!correct_symmetry)

    lattice = Matrix{T}(std_cell.lattice)
    positions = Vec3{T}.(std_cell.positions)
    magnetic_moments = normalize_magnetic_moment.(std_cell.magmoms)
    (; lattice, atom_groups, positions, magnetic_moments)
end
function spglib_standardize_cell(model, magnetic_moments=[]; kwargs...)
    spglib_standardize_cell(model.lattice, model.atom_groups, model.positions,
        magnetic_moments; kwargs...)
end


function spglib_spacegroup_number(model, magnetic_moments=[]; tol_symmetry=SYMMETRY_TOLERANCE)
    # Get spacegroup number according to International Tables for Crystallography (ITA)
    cell, _ = spglib_cell(model, magnetic_moments)
    Spglib.get_dataset(cell, tol_symmetry).spacegroup_number
end

"""
    standardize_cell(cell, symprec=1e-5; kwargs...)

Wrapper around `Spglib.standardize_cell` that returns a `StandardizedCell`,
providing error messages for deprecated v0.6 field names.
"""
function standardize_cell(cell, symprec=1e-5; kwargs...)
    StandardizedCell(Spglib.standardize_cell(cell, symprec; kwargs...))
end

"""
    get_ir_reciprocal_mesh(cell, mesh, is_shift; kwargs...)

Wrapper around `Spglib.get_ir_reciprocal_mesh` for BrillouinZoneMeshes.jl.
Returns a `Spglib.BrillouinZoneMesh` struct (v1.x API).
"""
function get_ir_reciprocal_mesh(cell, mesh, is_shift; kwargs...)
    Spglib.get_ir_reciprocal_mesh(cell, mesh; is_shift=is_shift, kwargs...)
end
