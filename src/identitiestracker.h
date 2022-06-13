#ifndef IDENTITIESTRACKER_H
#define IDENTITIESTRACKER_H
#include <map>
enum IdentityStatus {
    kUnidentified,
    kIdentified,
    kPending
};

struct Identity {
    IdentityStatus status{kPending};
};

class IdentitiesTracker
{   
public:
    bool Exists(long id) const;
    void Update(long id, IdentityStatus status);
    IdentityStatus GetStatus(long id) const;

private:
    std::map<long, Identity> data_;

};

#endif // IDENTITIESTRACKER_H
