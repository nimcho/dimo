"""
Adjustment Functions for BATS Dataset
"""

identity = lambda w: w

noun = lambda w: w + "-n"
adj = lambda w: w + "-j"
adv = lambda w: w + "-a"
verb = lambda w: w + "-v"

noun_cap = lambda w: noun(w.capitalize())
adj_cap = lambda w: adj(w.capitalize())

################################################################################

bats_conf = dict(

    #
    # Encyclopedic
    #

    e01=(noun_cap, noun_cap),  # capital -> country
    e02=(noun_cap, noun_cap),  # country -> language
    e03=(noun_cap, noun_cap),  # uk-city -> county
    e04=(noun_cap, adj_cap),  # surname -> nationality
    e05=(noun_cap, noun),  # surname -> occupation
    e06=(noun, noun),  # animal -> young
    e07=(noun, noun),  # animal -> sound
    # ... no, verbs do not work better for sounds

    e08=(noun, noun),  # animal -> shelter
    e09=(noun, adj),  # thing -> color
    # ... for colors, nouns works almost as well as adjs

    e10=(noun, noun),  # male -> female

    #
    # Lexicographic
    #

    # Note.
    # L07, L08 and L10 mix up POS tags,
    # so these 3 are manually edited

    l01=(noun, noun),  #
    l02=(noun, noun),  #
    l03=(noun, noun),  #
    l04=(noun, noun),  #
    l05=(noun, noun),  #
    l06=(noun, noun),  #
    l07=(identity, identity),  # synonyms - intensity
    l08=(identity, identity),  # synonyms - exact
    l09=(adj, adj),  # antonyms-gradable
    l10=(identity, identity),  # antonyms-binary

    #
    # Derivational
    #

    d01=(noun, adj),  # noun + <less>
    d02=(adj, adj),  # <un> + adj
    d03=(adj, adv),  # adj + <ly>
    d04=(adj, adj),  # <over> + adj
    d05=(adj, noun),  # adj + <ness>
    d06=(verb, verb),  # <re> + verb
    d07=(verb, adj),  # verb + <able>
    d08=(verb, noun),  # verb + <er>
    d09=(verb, noun),  # verb + <tion>
    d10=(verb, noun),  # verb + <ment>

    # ---
    # Note: Inflectional part is not here as SkEThes uses lemmas

)
