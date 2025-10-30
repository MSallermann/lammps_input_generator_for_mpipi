from mpipi_lammps_gen.globular_domains import (
    GlobularDomain,
    decide_globular_domains_from_sequence,
    merge_domains,
)


def test_globular_domain_criterion():
    threshold = 90.0
    minimum_domain_length = 3
    minimum_idr_length = 10

    plddt_low = threshold - 1.0
    plddt_high = threshold + 1.0

    IDR_short = [plddt_low] * (minimum_idr_length - 1)

    GLOB_short = [plddt_high] * (minimum_domain_length - 1)
    GLOB_long = [plddt_high] * (minimum_domain_length + 1)

    plddts = (
        IDR_short
        # This "domain" should be ignored because it's below the min domain length
        + GLOB_short
        + IDR_short
        # All of the following should be merged, because the IDRs are below the min IDR length
        + GLOB_long
        + IDR_short
        + GLOB_long
        + IDR_short
        + GLOB_long
        + IDR_short
        + GLOB_short  # This should be ignored
        + IDR_short
        + GLOB_long  # Should be separate from the others
    )

    domains = decide_globular_domains_from_sequence(
        plddts,
        threshold=threshold,
        minimum_domain_length=minimum_domain_length,
        minimum_idr_length=minimum_idr_length,
    )

    # little helper print to seee what's going on
    for idx, plddt in enumerate(plddts):
        if plddt < plddt_high:
            print(f"IDR: {idx}")
        else:
            print(f"GLO: {idx}")

    # two domains
    assert len(domains) == 2

    # the first one should have three fragments
    assert len(domains[0].indices) == 3

    assert domains[0].is_in_rigid_region(20)

    assert not domains[0].is_in_rigid_region(37)


def test_merge_domains():
    domains = [
        GlobularDomain([(0, 1)]),
        GlobularDomain([(2, 3)]),
        GlobularDomain([(4, 5)]),
        GlobularDomain([(6, 7)]),
        GlobularDomain([(8, 9)]),
    ]

    def should_be_merged(g1: GlobularDomain, g2: GlobularDomain):
        return (g1.start_idx() == 0 and g2.start_idx() <= 4) or (
            g1.start_idx() == 6 and g2.start_idx() == 8
        )

    new_domains = merge_domains(domains, should_be_merged)

    print(new_domains)

    assert len(new_domains) == 2
    assert new_domains[0].start_idx() == 0
    assert new_domains[0].end_idx() == 5
